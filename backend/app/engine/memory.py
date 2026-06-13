import logging
import math
import uuid
import re
import httpx
import json
from app.db.postgres import postgres_db
from app.db.qdrant import qdrant_db
from app.db.neo4j import neo4j_db
from app.utils.embeddings import get_ollama_embedding
from app.config import settings

logger = logging.getLogger("nexus.memory_agent")

class MemoryAgent:
    def __init__(self):
        pass

    async def extract_entities(self, statement: str) -> dict:
        """Invokes local Ollama chat API to map Project, Technology, and Concepts."""
        import time
        url = f"{settings.OLLAMA_URL}/api/chat"
        system_instruction = """
You are an entity extraction engine.

Return ONLY valid JSON.

Every key MUST be enclosed in double quotes.

Output format:

{
  "nodes": [
    {
      "name": "NoteHive",
      "label": "Project"
    }
  ],
  "relationships": [
    {
      "source": "NoteHive",
      "target": "FastAPI",
      "type": "BUILT_WITH"
    }
  ]
}

Do not output markdown.
Do not output explanations.
Do not output comments.
Return JSON only.
"""
        
        payload = {
            "model": settings.OLLAMA_REASONING_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Statement: \"{statement}\""}
            ],
            # We omit "format": "json" because reasoning models (like Qwen/DeepSeek reasoning models)
            # natively output <think>...</think> blocks, which are invalid JSON. Forcing JSON mode
            # causes them to stall, loop, or fail. We will parse the JSON manually instead.
            "options": {
                "temperature": 0.0
            },
            "stream": False
        }
        
        logger.info(
            f"Invoking Ollama chat at {url} for entity extraction.\n"
            f"Model: {settings.OLLAMA_REASONING_MODEL}\n"
            f"Statement: \"{statement}\""
        )
        
        start_time = time.time()
        try:
            # Using 90.0 seconds timeout to handle cold-starts or CPU-only inference on low-memory systems
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                content = data.get("message", {}).get("content", "")
                elapsed = time.time() - start_time
                logger.info(f"Ollama responded in {elapsed:.2f} seconds.")
                logger.debug(f"Raw Ollama output:\n{content}")
                
                if not content:
                    raise ValueError("Ollama response content was empty.")
                
                # 1. Strip think blocks if present
                parsed_text = content
                if "<think>" in parsed_text:
                    if "</think>" in parsed_text:
                        parts = parsed_text.split("</think>", 1)
                        think_content = parts[0].replace("<think>", "").strip()
                        logger.debug(f"Extracted reasoning/think block:\n{think_content}")
                        parsed_text = parts[1]
                    else:
                        parts = parsed_text.split("<think>", 1)
                        parsed_text = parts[0] # take before think
                
                # 2. Extract JSON substring between the first '{' and the last '}'
                start_idx = parsed_text.find("{")
                end_idx = parsed_text.rfind("}")
                
                if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                    logger.warning(f"Could not find JSON object boundary in text: {parsed_text}")
                    raise ValueError("No valid JSON structure found in LLM response.")
                
                json_str = parsed_text[start_idx : end_idx + 1]
                logger.debug(f"Extracted JSON string for parsing: {json_str}")
        
                logger.info(f"RAW OLLAMA RESPONSE:\n{json_str}")

                try:
                    extracted = json.loads(json_str)

                except json.JSONDecodeError:
                    logger.warning("Attempting JSON repair...")

                    json_str = re.sub(
                        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
                        r'\1"\2"\3',
                        json_str
                    )

                    logger.info(f"REPAIRED JSON:\n{json_str}")

                    extracted = json.loads(json_str)
                
                # Ensure list structures exist
                if "nodes" not in extracted:
                    extracted["nodes"] = []
                if "relationships" not in extracted:
                    extracted["relationships"] = []
                
                logger.info(
                    f"Successfully extracted {len(extracted['nodes'])} nodes "
                    f"and {len(extracted['relationships'])} relationships."
                )
                return extracted
                
        except Exception as e:
            logger.error(f"Failed to extract entities from Ollama: {e}", exc_info=True)
            # Safe default fallback for MVP to prevent full ingestion crash
            return {"nodes": [], "relationships": []}

    async def store_memory(self, statement: str, category: str = "general") -> dict:
        """Memory-Centric coordinated ingestion pipeline with full database transactional rollbacks."""
        logger.info(f"Storing new memory statement: '{statement[:50]}...'")
        
        user_id = await postgres_db.get_or_create_user()
        belief_id = None
        vector_stored = False
        graph_nodes_created = False
        
        try:
            # 1. PostgreSQL raw log event
            belief_id = await postgres_db.create_episodic_log(
                user_id=user_id,
                event_type="INPUT",
                statement=statement,
                metadata={"category": category}
            )
            
            # 2. Generate vector embedding
            vector = await get_ollama_embedding(statement)
            
            # 3. Qdrant vector upsert
            await qdrant_db.upsert_memory_vector(
                belief_id=belief_id,
                vector=vector,
                statement=statement,
                user_id=user_id
            )
            vector_stored = True
            
            # 4. Ollama Entity/Relations Extraction
            extracted = await self.extract_entities(statement)
            
            # 5. Neo4j Memory-Centric Graph writes
            # Phase A: Create parent Memory Node linked to User
            await neo4j_db.create_memory_node(
                belief_id=belief_id,
                statement=statement,
                user_id=user_id
            )
            graph_nodes_created = True
            
            # Phase B: Create Concept, Project, Tech nodes linked to Memory Node
            await neo4j_db.create_semantic_entities(
                belief_id=belief_id,
                nodes=extracted.get("nodes", []),
                relationships=extracted.get("relationships", [])
            )
            
            # 6. Contradiction Detection Sweep
            from app.engine.conflict import conflict_agent
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
                "conflict_flagged": conflict_flagged
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
