import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Mock database connections on startup
with patch('app.db.postgres.postgres_db.connect', AsyncMock()), \
     patch('app.db.neo4j.neo4j_db.connect', AsyncMock()), \
     patch('app.db.qdrant.qdrant_db.connect', AsyncMock()):
    from main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_get_dashboard_stats(client):
    mock_stats = {
        "projects": 3,
        "technologies": 5,
        "careers": 2,
        "contradictions": 1
    }
    with patch("app.db.postgres.postgres_db.get_memories_count", AsyncMock(return_value=12)) as mock_count, \
         patch("app.db.neo4j.neo4j_db.get_dashboard_stats", AsyncMock(return_value=mock_stats)) as mock_neo_stats:
        response = client.get("/api/v1/dashboard/stats")
        assert response.status_code == 200
        assert response.json() == {
            "memories": 12,
            "projects": 3,
            "technologies": 5,
            "careers": 2,
            "contradictions": 1
        }
        mock_count.assert_called_once()
        mock_neo_stats.assert_called_once()

def test_get_dashboard_memories(client):
    mock_memories = [
        {
            "belief_id": "11111111-1111-1111-1111-111111111111",
            "statement": "I love FastAPI",
            "timestamp": "2026-06-14T00:00:00+00:00",
            "event_type": "user_statement"
        }
    ]
    with patch("app.db.postgres.postgres_db.get_latest_memories", AsyncMock(return_value=mock_memories)) as mock_get:
        response = client.get("/api/v1/dashboard/memories?limit=10")
        assert response.status_code == 200
        assert response.json() == mock_memories
        mock_get.assert_called_once_with(limit=10)

def test_get_dashboard_projects(client):
    mock_projects = [
        {"name": "Nexus", "id": "p1", "created_at": 1718323200}
    ]
    with patch("app.db.neo4j.neo4j_db.get_all_projects", AsyncMock(return_value=mock_projects)) as mock_get:
        response = client.get("/api/v1/dashboard/projects")
        assert response.status_code == 200
        assert response.json() == mock_projects
        mock_get.assert_called_once()

def test_get_dashboard_technologies(client):
    mock_techs = [
        {"name": "FastAPI", "id": "t1"}
    ]
    with patch("app.db.neo4j.neo4j_db.get_all_technologies", AsyncMock(return_value=mock_techs)) as mock_get:
        response = client.get("/api/v1/dashboard/technologies")
        assert response.status_code == 200
        assert response.json() == [
            {"name": "FastAPI", "id": "t1", "created_at": None}
        ]
        mock_get.assert_called_once()

def test_get_dashboard_careers(client):
    mock_careers = [
        {"name": "AI Engineer", "id": "c1"}
    ]
    with patch("app.db.neo4j.neo4j_db.get_all_careers", AsyncMock(return_value=mock_careers)) as mock_get:
        response = client.get("/api/v1/dashboard/careers")
        assert response.status_code == 200
        assert response.json() == [
            {"name": "AI Engineer", "id": "c1", "created_at": None}
        ]
        mock_get.assert_called_once()

def test_get_dashboard_contradictions(client):
    mock_contradictions = [
        {
            "id": "cr1",
            "category": "career",
            "status": "active",
            "severity": "high",
            "created_at": 1718323200
        }
    ]
    with patch("app.db.neo4j.neo4j_db.get_all_contradictions", AsyncMock(return_value=mock_contradictions)) as mock_get:
        response = client.get("/api/v1/dashboard/contradictions")
        assert response.status_code == 200
        assert response.json() == mock_contradictions
        mock_get.assert_called_once()

def test_get_dashboard_graph(client):
    mock_graph = {
        "nodes": [
            {"id": "n1", "name": "Nexus", "label": "Project"},
            {"id": "n2", "name": "FastAPI", "label": "Technology"}
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "BUILT_WITH"}
        ]
    }
    with patch("app.db.neo4j.neo4j_db.get_dashboard_graph", AsyncMock(return_value=mock_graph)) as mock_get:
        response = client.get("/api/v1/dashboard/graph")
        assert response.status_code == 200
        assert response.json() == mock_graph
        mock_get.assert_called_once()
