import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Mock database connections on startup
with patch('app.db.postgres.postgres_db.connect', AsyncMock()), \
     patch('app.db.neo4j.neo4j_db.connect', AsyncMock()), \
     patch('app.db.qdrant.qdrant_db.connect', AsyncMock()):
    from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_conflict_agent_detection():
    from app.engine.conflict import conflict_agent
    from app.db.postgres import postgres_db
    from app.db.qdrant import qdrant_db
    from app.db.neo4j import neo4j_db
    
    belief_id = "11111111-1111-1111-1111-111111111111"
    user_id = "00000000-0000-0000-0000-000000000000"
    statement = "I want to focus on backend engineering"
    
    mock_vector = [0.1] * 384
    mock_hits = [
        {
            "belief_id": "22222222-2222-2222-2222-222222222222",
            "statement": "I want to become an ML engineer",
            "similarity": 0.85,
            "confidence": 1.0
        }
    ]
    
    # Mock embeddings and database interactions
    with patch("app.engine.conflict.get_ollama_embedding", AsyncMock(return_value=mock_vector)), \
         patch.object(qdrant_db, "search_similar_memories", AsyncMock(return_value=mock_hits)), \
         patch.object(conflict_agent, "verify_dissonance", AsyncMock(return_value={
             "is_contradiction": True,
             "category": "GOAL",
             "explanation": "Backend engineering vs ML engineering goal conflict."
         })), \
         patch.object(neo4j_db, "get_concept_centrality", AsyncMock(return_value=0.5)), \
         patch.object(postgres_db, "create_contradiction_log", AsyncMock(return_value="33333333-3333-3333-3333-333333333333")) as mock_pg_log, \
         patch.object(neo4j_db, "create_contradiction_node", AsyncMock()) as mock_neo_node, \
         patch.object(neo4j_db, "decay_recall_confidence", AsyncMock()) as mock_decay:
         
        conflict_flagged = await conflict_agent.check_contradictions(belief_id, statement, user_id)
        
        assert conflict_flagged is True
        mock_pg_log.assert_called_once_with(
            user_id=user_id,
            concept_a_id=belief_id,
            concept_b_id="22222222-2222-2222-2222-222222222222",
            severity="CRITICAL" # similarity 0.85 * category weight 1.0 (GOAL) * (centrality 0.5 + 0.5) = 0.85 >= 0.75 -> CRITICAL
        )
        mock_neo_node.assert_called_once()
        mock_decay.assert_called_once()

@pytest.mark.asyncio
async def test_conflict_agent_resolution():
    from app.engine.conflict import conflict_agent
    from app.db.postgres import postgres_db
    from app.db.neo4j import neo4j_db
    
    contradiction_id = "33333333-3333-3333-3333-333333333333"
    keep_belief_id = "11111111-1111-1111-1111-111111111111"
    supersede_belief_id = "22222222-2222-2222-2222-222222222222"
    transition_trigger = "User manually decided to focus on backend"
    
    with patch.object(postgres_db, "resolve_contradiction_log", AsyncMock()) as mock_pg_resolve, \
         patch.object(neo4j_db, "resolve_contradiction_graph", AsyncMock()) as mock_neo_resolve:
         
        await conflict_agent.resolve_contradiction(
            contradiction_id=contradiction_id,
            keep_belief_id=keep_belief_id,
            supersede_belief_id=supersede_belief_id,
            transition_trigger=transition_trigger
        )
        
        mock_pg_resolve.assert_called_once()
        mock_neo_resolve.assert_called_once_with(
            contradiction_id=contradiction_id,
            keep_belief_id=keep_belief_id,
            supersede_belief_id=supersede_belief_id,
            transition_trigger=transition_trigger
        )

def test_api_resolve_endpoint(client):
    payload = {
        "conflict_id": "33333333-3333-3333-3333-333333333333",
        "keep_belief_id": "11111111-1111-1111-1111-111111111111",
        "supersede_belief_id": "22222222-2222-2222-2222-222222222222",
        "transition_trigger": "User selection"
    }
    
    with patch("app.engine.conflict.conflict_agent.resolve_contradiction", AsyncMock()) as mock_resolve:
        response = client.post("/api/v1/conflict/resolve", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "status": "resolved",
            "message": "Contradiction 33333333-3333-3333-3333-333333333333 resolved successfully."
        }
        mock_resolve.assert_called_once_with(
            contradiction_id=payload["conflict_id"],
            keep_belief_id=payload["keep_belief_id"],
            supersede_belief_id=payload["supersede_belief_id"],
            transition_trigger=payload["transition_trigger"]
        )
