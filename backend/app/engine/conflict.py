import logging
import json
import httpx
from app.db.postgres import postgres_db
from app.db.qdrant import qdrant_db
from app.db.neo4j import neo4j_db
from app.utils.embeddings import get_ollama_embedding
from app.config import settings

logger = logging.getLogger("nexus.conflict_agent")

class ConflictAgent:
    def __init__(self):
        pass

    async def verify_dissonance(self, statement_a: str, statement_b: str) -> dict:
        """Invokes local Ollama API to determine if a logical contradiction exists with robust parsing."""
        url = f"{settings.OLLAMA_URL}/api/chat"
        system_instruction = (
            "You are a logical contradiction detection engine.\n\n"
            "Compare Statement A and Statement B and determine if they contain a logical contradiction (either direct or contextual).\n"
            "A contradiction exists if Statement A and Statement B express mutually exclusive states, goals, preferences, or choices for the same user.\n\n"
            "Apply these explicit contradiction rules:\n"
            "1. Career: Desiring mutually exclusive career roles or paths at the same time (e.g., wanting to become a 'Backend Engineer' vs a 'Doctor').\n"
            "2. Education: Pursuing conflicting full-time academic fields or degrees concurrently (e.g., studying for a 'BTech' vs an 'MBBS').\n"
            "3. Preferences: Directly conflicting personal or professional preferences (e.g., 'remote' vs 'on-site' work preferences).\n"
            "4. Technology: Conflicting technology choices or stack preferences for the same scenario (e.g., 'React' vs 'Angular' preference, or 'SQL' vs 'NoSQL' database preferences).\n\n"
            "Categorize the contradiction into: PREFERENCE, TECH_PREFERENCE, GOAL, KNOWLEDGE.\n"
            "Return ONLY a JSON payload matching this schema (no markdown formatting, no explanations, no comments):\n"
            "{\n"
            "  \"is_contradiction\": true|false,\n"
            "  \"category\": \"PREFERENCE|TECH_PREFERENCE|GOAL|KNOWLEDGE\",\n"
            "  \"explanation\": \"Brief explanation of the logical conflict\"\n"
            "}"
        )
        
        payload = {
            "model": settings.OLLAMA_REASONING_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Statement A: \"{statement_a}\"\nStatement B: \"{statement_b}\""}
            ],
            "options": {
                "temperature": 0.0
            },
            "stream": False
        }
        
        # Log the full Ollama payload
        logger.info(f"Ollama contradiction reasoning request payload:\n{json.dumps(payload, indent=2)}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                
                # Log the full raw response
                logger.info(f"Ollama contradiction reasoning raw response content:\n{content}")
                
                if not content:
                    raise ValueError("Ollama response content was empty.")
                
                # 1. Strip think blocks if present
                parsed_text = content
                if "<think>" in parsed_text:
                    if "</think>" in parsed_text:
                        parts = parsed_text.split("</think>", 1)
                        parsed_text = parts[1]
                    else:
                        parts = parsed_text.split("<think>", 1)
                        parsed_text = parts[0]
                
                # 2. Extract JSON substring
                start_idx = parsed_text.find("{")
                end_idx = parsed_text.rfind("}")
                if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                    raise ValueError("Could not find JSON object in Ollama response.")
                
                json_str = parsed_text[start_idx : end_idx + 1]
                
                # Attempt to parse JSON
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("Attempting JSON repair for contradiction response...")
                    import re
                    json_str = re.sub(
                        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
                        r'\1"\2"\3',
                        json_str
                    )
                    result = json.loads(json_str)
                
                logger.info(f"Parsed contradiction decision: {result}")
                return {
                    "is_contradiction": bool(result.get("is_contradiction", False)),
                    "category": result.get("category", "PREFERENCE"),
                    "explanation": result.get("explanation", "")
                }
        except Exception as e:
            logger.error(f"Failed to check contradiction via Ollama: {e}", exc_info=True)
            return {"is_contradiction": False, "category": "PREFERENCE", "explanation": f"Error during parsing: {str(e)}"}

    def _calculate_severity(self, similarity: float, category: str, centrality: float) -> tuple[float, str]:
        """Calculates severity score and assigns severity classification label."""
        category_weights = {
            "GOAL": 1.0,
            "KNOWLEDGE": 0.8,
            "TECH_PREFERENCE": 0.6,
            "PREFERENCE": 0.4
        }
        w_category = category_weights.get(category, 0.4)
        
        # Severity equation
        score = similarity * w_category * (centrality + 0.5)  # Offset centrality slightly to prevent zeroing
        score = min(max(score, 0.0), 1.0)
        
        if score >= 0.75:
            return score, "CRITICAL"
        elif score >= 0.50:
            return score, "HIGH"
        elif score >= 0.25:
            return score, "MEDIUM"
        else:
            return score, "LOW"

    async def check_contradictions(self, belief_id: str, statement: str, user_id: str) -> bool:
        """Scans Qdrant and Neo4j for logical contradictions and logs active conflicts."""
        logger.info(f"Scanning for contradictions matching statement: '{statement[:50]}'")
        
        # 1. Retrieve semantic neighbors
        query_vector = await get_ollama_embedding(statement)
        vector_hits = await qdrant_db.search_similar_memories(query_vector, limit=5)
        
        logger.info(f"Retrieved {len(vector_hits)} memories from Qdrant for contradiction detection.")
        for idx, hit in enumerate(vector_hits):
            logger.info(
                f"  Hit [{idx}]: belief_id={hit['belief_id']}, "
                f"similarity={hit['similarity']:.4f}, "
                f"statement='{hit['statement']}'"
            )
        
        conflict_flagged = False
        for hit in vector_hits:
            target_belief_id = hit["belief_id"]
            if target_belief_id == belief_id:
                continue
                
            # Compute cosine similarity
            similarity = hit["similarity"]
            # Threshold lowered to 0.60 to capture semantic conflicts in similar context sentences
            if similarity < 0.60:
                logger.info(f"Skipping hit '{target_belief_id}' due to low similarity ({similarity:.4f} < 0.60)")
                continue
                
            # 2. Invoke LLM validation checks
            dissonance = await self.verify_dissonance(statement, hit["statement"])
            if dissonance["is_contradiction"]:
                logger.warning(f"Logical contradiction detected! Category: {dissonance['category']}. Reason: {dissonance['explanation']}")
                
                # Fetch centrality to compute severity
                centrality_a = await neo4j_db.get_concept_centrality(belief_id)
                centrality_b = await neo4j_db.get_concept_centrality(target_belief_id)
                avg_centrality = (centrality_a + centrality_b) / 2.0
                
                # Calculate severity
                severity_score, severity_label = self._calculate_severity(
                    similarity=similarity,
                    category=dissonance["category"],
                    centrality=avg_centrality
                )
                
                # 3. PostgreSQL log entry
                contradiction_id = await postgres_db.create_contradiction_log(
                    user_id=user_id,
                    concept_a_id=belief_id,
                    concept_b_id=target_belief_id,
                    severity=severity_label
                )
                
                # 4. Neo4j contradiction junction node creation
                await neo4j_db.create_contradiction_node(
                    contradiction_id=contradiction_id,
                    category=dissonance["category"],
                    severity=severity_label,
                    belief_ids=[belief_id, target_belief_id]
                )
                
                # 5. Decay recall confidence weights
                await neo4j_db.decay_recall_confidence(
                    belief_ids=[belief_id, target_belief_id],
                    severity_score=severity_score
                )
                
                conflict_flagged = True
                
        return conflict_flagged

    async def resolve_contradiction(
        self, contradiction_id: str, keep_belief_id: str, supersede_belief_id: str, transition_trigger: str
    ):
        """Executes automated or user-assisted database revisions to clear a conflict state."""
        logger.info(f"Resolving contradiction '{contradiction_id}' keeping: {keep_belief_id}")
        
        # 1. Update PostgreSQL resolution logs
        details = f"Resolved via user selection override. Kept memory: {keep_belief_id}. Archived: {supersede_belief_id}. Trigger: {transition_trigger}"
        await postgres_db.resolve_contradiction_log(contradiction_id, details)
        
        # 2. Resolve Neo4j nodes and map SUPERSEDES link
        await neo4j_db.resolve_contradiction_graph(
            contradiction_id=contradiction_id,
            keep_belief_id=keep_belief_id,
            supersede_belief_id=supersede_belief_id,
            transition_trigger=transition_trigger
        )
        logger.info(f"Contradiction resolution complete for: {contradiction_id}")

# Singleton instance
conflict_agent = ConflictAgent()
