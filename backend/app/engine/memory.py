import logging
import math
import uuid
import re
import httpx
import json
import time
from app.db.postgres import postgres_db
from app.db.qdrant import qdrant_db
from app.db.neo4j import neo4j_db
from app.utils.embeddings import get_ollama_embedding
from app.config import settings
from app.engine.conflict import conflict_agent

logger = logging.getLogger("nexus.memory_agent")

class MemoryAgent:
    def __init__(self):
        pass

    async def extract_entities(self, statement: str) -> dict:
        """Invokes local Ollama chat API to map Project, Technology, and Concepts."""
        url = f"{settings.OLLAMA_URL}/api/chat"
        system_instruction = (
            "You are a strict entity and relationship extraction engine.\n\n"
            "CRITICAL RULES:\n"
            "1. Extract entities and relationships ONLY from the supplied statement.\n"
            "2. Do not use prior memories.\n"
            "3. Do not use retrieved context.\n"
            "4. Do not use examples or infer entities not explicitly mentioned in the statement.\n"
            "5. Do not infer missing entities.\n\n"
            "Allowed labels for entities:\n"
            "- Project: Software projects, apps, products, or systems.\n"
            "- Technology: Languages, frameworks, databases, libraries, tools, or platforms.\n"
            "- Skill: Specific capabilities or areas of expertise.\n"
            "- Career: Job roles, occupations, or titles.\n"
            "- Company: Organizations or employers.\n"
            "- Person: Individual names.\n"
            "- Concept: General abstract ideas or domains not matching other labels.\n\n"
            "Forbidden labels:\n"
            "- Subject, Verb, Object, Determiner, Preposition\n\n"
            "Forbidden entities (NEVER extract these):\n"
            "- I, want, to, a, the, pronouns, or common grammatical stopwords.\n\n"
            "Return ONLY valid JSON matching this structure (no markdown, no explanations, no comments):\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\n"
            "      \"name\": \"<extracted entity name>\",\n"
            "      \"label\": \"Project|Technology|Skill|Career|Company|Person|Concept\"\n"
            "    }\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\n"
            "      \"source\": \"<source entity name>\",\n"
            "      \"target\": \"<target entity name>\",\n"
            "      \"type\": \"BUILT_WITH|USES|HAS_SKILL|WORKS_AT|WANTS_CAREER|RELATED_TO\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        payload = {
            "model": settings.OLLAMA_REASONING_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Statement: \"{statement}\""}
            ],
            "options": {
                "temperature": 0.0
            },
            "stream": False
        }

        # Log the full Ollama payload
        logger.info(f"Ollama entity extraction chat request payload:\n{json.dumps(payload, indent=2)}")

        def validate_schema(data: dict) -> bool:
            if not isinstance(data, dict):
                return False
            if "nodes" not in data or not isinstance(data["nodes"], list):
                return False
            if "relationships" not in data or not isinstance(data["relationships"], list):
                return False
            for node in data["nodes"]:
                if not isinstance(node, dict) or "name" not in node or "label" not in node:
                    return False
            for rel in data["relationships"]:
                if not isinstance(rel, dict) or "source" not in rel or "target" not in rel or "type" not in rel:
                    return False
            return True

        def regex_fallback_parse(text: str) -> dict:
            nodes = []
            relationships = []
            # Find name/label objects
            node_matches = re.findall(
                r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}',
                text
            )
            for name, label in node_matches:
                nodes.append({"name": name, "label": label})
            
            # Find source/target/type objects
            rel_matches = re.findall(
                r'\{\s*"source"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*\}',
                text
            )
            for src, tgt, rtype in rel_matches:
                relationships.append({"source": src, "target": tgt, "type": rtype})
            
            return {"nodes": nodes, "relationships": relationships}

        extracted = {"nodes": [], "relationships": []}
        max_attempts = 2
        success = False

        for attempt in range(1, max_attempts + 1):
            attempt_start = time.time()
            logger.info(f"Ollama entity extraction attempt number: {attempt}")
            try:
                # Using 90.0 seconds timeout
                async with httpx.AsyncClient(timeout=90.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    content = data.get("message", {}).get("content", "")
                    duration_ms = int((time.time() - attempt_start) * 1000)
                    logger.info(f"Extraction attempt {attempt} response received in {duration_ms} ms.")
                    logger.info(f"Raw Ollama response content:\n{content}")
                    
                    if not content:
                        raise ValueError("Ollama response content was empty.")
                    
                    # 1. Strip think blocks
                    parsed_text = content
                    if "<think>" in parsed_text:
                        if "</think>" in parsed_text:
                            parts = parsed_text.split("</think>", 1)
                            think_content = parts[0].replace("<think>", "").strip()
                            logger.debug(f"Extracted reasoning/think block:\n{think_content}")
                            parsed_text = parts[1]
                        else:
                            parts = parsed_text.split("<think>", 1)
                            parsed_text = parts[0]
                    
                    # 2. Extract JSON substring
                    start_idx = parsed_text.find("{")
                    end_idx = parsed_text.rfind("}")
                    
                    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                        raise ValueError("No JSON boundaries found.")
                    
                    json_str = parsed_text[start_idx : end_idx + 1]
                    
                    # Attempt JSON repair if needed
                    try:
                        extracted = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning("Attempting JSON repair...")
                        json_str = re.sub(
                            r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
                            r'\1"\2"\3',
                            json_str
                        )
                        extracted = json.loads(json_str)
                    
                    if validate_schema(extracted):
                        success = True
                        break
                    else:
                        raise ValueError("Parsed JSON violates the extraction schema.")
                        
            except Exception as e:
                duration_ms = int((time.time() - attempt_start) * 1000)
                logger.warning(f"Attempt {attempt} failed: {e}. Duration: {duration_ms} ms")
                # If this is the last attempt, fall back to regex parsing
                if attempt == max_attempts:
                    logger.warning("All extraction attempts failed schema validation. Falling back to regex parsing.")
                    extracted = regex_fallback_parse(content if 'content' in locals() else "")
                    success = True

        # Post-processing Grounding Validation Layer
        valid_nodes = []
        valid_node_names = set()
        forbidden_words = {"i", "want", "to", "a", "the"}
        forbidden_labels = {"subject", "verb", "object", "determiner", "preposition"}

        for node in extracted.get("nodes", []):
            name = node.get("name")
            label = node.get("label")
            if not name or not label:
                continue

            name_lower = name.strip().lower()
            label_lower = label.strip().lower()

            # Rule: entity_name.lower() must exist inside statement.lower()
            if name_lower not in statement.lower():
                logger.warning(
                    f"GROUNDING VIOLATION: Discarding hallucinated entity node '{name}' "
                    f"because it does not appear in the statement: '{statement}'"
                )
                continue

            # Rule: no forbidden entities
            if name_lower in forbidden_words:
                logger.warning(f"GROUNDING VIOLATION: Discarding forbidden entity word '{name}'")
                continue

            # Rule: no forbidden labels
            if label_lower in forbidden_labels:
                logger.warning(f"GROUNDING VIOLATION: Discarding forbidden label '{label}' on entity '{name}'")
                continue

            valid_nodes.append(node)
            valid_node_names.add(name_lower)

        # Remove relationships whose endpoints were discarded or don't match the statement
        valid_rels = []
        for rel in extracted.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            if not source or not target or not rel_type:
                continue

            source_lower = source.strip().lower()
            target_lower = target.strip().lower()

            if source_lower in valid_node_names and target_lower in valid_node_names:
                valid_rels.append(rel)
            else:
                logger.warning(
                    f"GROUNDING VIOLATION: Discarding relationship ({source})-[:{rel_type}]->({target}) "
                    f"because one or both endpoints were not valid extracted entities."
                )

        validated_entities = {
            "nodes": valid_nodes,
            "relationships": valid_rels
        }
        logger.info(f"Validated entity list after grounding check:\n{json.dumps(validated_entities, indent=2)}")
        return validated_entities

    async def store_memory(self, statement: str, category: str = "general") -> dict:
        """Memory-Centric coordinated ingestion pipeline with duplicate prevention and database transaction control."""
        logger.info(f"Storing new memory statement: '{statement[:50]}...'")
        
        user_id = await postgres_db.get_or_create_user()
        belief_id = None
        vector_stored = False
        graph_nodes_created = False
        
        try:
            # 1. Generate vector embedding first
            vector = await get_ollama_embedding(statement)
            
            # 2. Check Qdrant for duplicate memories (similarity > 0.95)
            similar_memories = await qdrant_db.search_similar_memories(vector, limit=1)
            if similar_memories:
                top_hit = similar_memories[0]
                similarity = top_hit["similarity"]
                logger.info(f"Duplicate check: top hit similarity is {similarity:.4f} for '{top_hit['statement'][:30]}...'")
                
                if similarity > 0.95:
                    existing_belief_id = top_hit["belief_id"]
                    logger.info(f"DUPLICATE DETECTED (similarity {similarity:.4f} > 0.95) with belief '{existing_belief_id}'. Reinforcing access metadata.")
                    
                    # Update Postgres access recall log
                    await postgres_db.log_memory_recall(
                        belief_id=existing_belief_id,
                        retrieved_score=1.0,
                        reinforced_strength=1.2
                    )
                    
                    # Update Neo4j access recalls edge
                    await neo4j_db.update_memory_access(
                        belief_id=existing_belief_id,
                        user_id=user_id
                    )
                    
                    return {
                        "status": "duplicate_detected",
                        "belief_id": existing_belief_id,
                        "user_id": user_id,
                        "extracted_entities": {"nodes": [], "relationships": []},
                        "conflict_flagged": False,
                        "graph_stats": {
                            "nodes_processed": 0,
                            "nodes_created_db": 0,
                            "relationships_processed": 0,
                            "relationships_created_db": 0
                        }
                    }

            # 3. PostgreSQL raw log event (unique statement)
            belief_id = await postgres_db.create_episodic_log(
                user_id=user_id,
                event_type="INPUT",
                statement=statement,
                metadata={"category": category}
            )
            
            # 4. Qdrant vector upsert
            await qdrant_db.upsert_memory_vector(
                belief_id=belief_id,
                vector=vector,
                statement=statement,
                user_id=user_id
            )
            vector_stored = True
            
            # 5. Ollama Entity/Relations Extraction
            extracted = await self.extract_entities(statement)
            
            # 6. Neo4j Memory-Centric Graph writes
            # Phase A: Create parent Memory Node linked to User
            await neo4j_db.create_memory_node(
                belief_id=belief_id,
                statement=statement,
                user_id=user_id
            )
            graph_nodes_created = True
            
            # Phase B: Create Concept, Project, Tech nodes linked to Memory Node
            graph_stats = await neo4j_db.create_semantic_entities(
                belief_id=belief_id,
                nodes=extracted.get("nodes", []),
                relationships=extracted.get("relationships", [])
            )
            
            # 7. Contradiction Detection Sweep
            conflict_flagged = await conflict_agent.check_contradictions(
                belief_id=belief_id,
                statement=statement,
                user_id=user_id
            )
            
            return {
                "status": "stored",
                "belief_id": belief_id,
                "user_id": user_id,
                "extracted_entities": extracted,
                "conflict_flagged": conflict_flagged,
                "graph_stats": graph_stats
            }
            
        except Exception as e:
            logger.error(f"Transaction failed during ingestion: {e}. Initiating rollback...")
            # Trigger sequential rollbacks for written database rows
            if graph_nodes_created and belief_id:
                try:
                    await neo4j_db.delete_memory_nodes(belief_id)
                except Exception as ne:
                    logger.error(f"Failed to rollback Neo4j memory nodes: {ne}")
            
            if vector_stored and belief_id:
                try:
                    await qdrant_db.delete_memory_vector(belief_id)
                except Exception as qe:
                    logger.error(f"Failed to rollback Qdrant vector point: {qe}")
                    
            if belief_id:
                try:
                    await postgres_db.delete_episodic_log(belief_id)
                except Exception as pe:
                    logger.error(f"Failed to rollback PostgreSQL episodic log: {pe}")
            raise e


    async def retrieve_memories(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid memory-centric retrieval querying Qdrant, Neo4j, and Postgres with RFSC-M ranking."""
        logger.info(f"Retrieving context-aware memories for query: '{query}'")
        
        # 1. Get query embedding
        query_vector = await get_ollama_embedding(query)
        
        # 2. Search Qdrant
        vector_hits = await qdrant_db.search_similar_memories(query_vector, limit=limit)
        if not vector_hits:
            return []
            
        belief_ids = [hit["belief_id"] for hit in vector_hits]
        
        # 3. Asynchronously fetch Neo4j memory context and Postgres metadata
        graph_context_map = await neo4j_db.retrieve_memory_graph_context(belief_ids)
        metadata_map = await postgres_db.get_memory_metadata(belief_ids)
        
        ranked_memories = []
        for hit in vector_hits:
            belief_id = hit["belief_id"]
            
            # Fetch Neo4j graph details
            graph_data = graph_context_map.get(belief_id, {"entities": [], "neighbors": []})
            entities = graph_data["entities"]
            neighbors = graph_data["neighbors"]
            
            # Fetch temporal decay parameters from Postgres
            metadata = metadata_map.get(belief_id, {"elapsed_seconds": 0.0, "frequency": 1})
            elapsed_days = metadata["elapsed_seconds"] / 86400.0  # Scale elapsed time to days
            frequency = metadata["frequency"]
            
            # 4. Fetch PageRank/Degree centrality of the Memory node cluster
            centrality = await neo4j_db.get_concept_centrality(belief_id)
            
            # 5. Calculate Retrievability decay score (R)
            lambda_0 = 0.05  # Base semantic decay rate per day
            alpha = 0.1
            beta = 0.2
            confidence = hit["confidence"]
            
            # Dynamic decay coefficient
            decay_lambda = lambda_0 / (1.0 + alpha * math.log1p(frequency) + beta * confidence)
            retrievability = math.exp(-decay_lambda * elapsed_days)
            
            # 6. Calculate composite RFSC-M Score
            w_similarity = 0.4
            w_retrievability = 0.3
            w_centrality = 0.2
            w_confidence = 0.1
            
            rfsc_score = (
                w_similarity * hit["similarity"] +
                w_retrievability * retrievability +
                w_centrality * centrality +
                w_confidence * confidence
            )
            
            # 7. Log recall access event (Retrieval Reinforcement)
            strength_multiplier = 1.0 + 0.2 * (1.0 - retrievability)
            await postgres_db.log_memory_recall(
                belief_id=belief_id,
                retrieved_score=rfsc_score,
                reinforced_strength=strength_multiplier
            )
            
            ranked_memories.append({
                "belief_id": belief_id,
                "statement": hit["statement"],
                "rfsc_score": round(rfsc_score, 3),
                "provenance": {
                    "similarity": round(hit["similarity"], 3),
                    "retrievability": round(retrievability, 3),
                    "last_accessed_days_ago": round(elapsed_days, 3),
                    "access_frequency": frequency,
                    "confidence": confidence
                },
                "extracted_entities": entities,
                "graph_neighbors": neighbors
            })
            
        # Sort by composite RFSC-M score descending
        ranked_memories.sort(key=lambda x: x["rfsc_score"], reverse=True)
        return ranked_memories


    async def get_all_memories_with_decay(self) -> list[dict]:
        """Lists active memories and calculates retrievability for frontend visualization."""
        user_id = await postgres_db.get_or_create_user()
        
        # Fetch event logs
        if postgres_db.pool is None:
            return []
            
        async with postgres_db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, statement, timestamp,
                       (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - timestamp))) AS elapsed_seconds
                FROM cognitive_events
                WHERE user_id = $1
                ORDER BY timestamp DESC
                """,
                uuid.UUID(user_id)
            )
            
        memories = []
        for row in rows:
            belief_id = str(row["id"])
            elapsed_days = float(row["elapsed_seconds"]) / 86400.0
            
            # Simple decay calculation for list dashboard
            lambda_0 = 0.05
            retrievability = math.exp(-lambda_0 * elapsed_days)
            
            memories.append({
                "belief_id": belief_id,
                "statement": row["statement"],
                "timestamp": row["timestamp"].isoformat(),
                "retrievability": round(retrievability, 3)
            })
        return memories

# Singleton instance
memory_agent = MemoryAgent()
