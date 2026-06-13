import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.engine.memory import memory_agent
from app.engine.conflict import conflict_agent
from app.db.postgres import postgres_db
from app.db.neo4j import neo4j_db

logger = logging.getLogger("nexus.endpoints")
router = APIRouter()

# Pydantic Schemas
class StoreMemoryRequest(BaseModel):
    statement: str = Field(..., description="The semantic text statement to commit to memory.")
    category: str = Field("general", description="The contextual category classification of the memory.")

class StoreMemoryResponse(BaseModel):
    status: str
    belief_id: str
    user_id: str
    extracted_entities: dict = Field(default_factory=dict)
    conflict_flagged: bool = Field(default=False)
    graph_stats: dict = Field(default_factory=dict)

class ResolveContradictionRequest(BaseModel):
    conflict_id: str = Field(..., description="The UUID of the contradiction log/node.")
    keep_belief_id: str = Field(..., description="The ID of the memory statement to retain as active.")
    supersede_belief_id: str = Field(..., description="The ID of the memory statement to supersede/deactivate.")
    transition_trigger: str = Field(..., description="The rationale or transition event triggering the resolution.")

class ResolveContradictionResponse(BaseModel):
    status: str
    message: str

class MemoryProvenance(BaseModel):
    similarity: float
    retrievability: float
    last_accessed_days_ago: float
    access_frequency: int
    confidence: float

class ExtractedEntity(BaseModel):
    name: str
    label: str

class RetrieveMemoryResponse(BaseModel):
    belief_id: str
    statement: str
    rfsc_score: float
    provenance: MemoryProvenance
    extracted_entities: list[ExtractedEntity]
    graph_neighbors: list[str]


class DecayLogResponse(BaseModel):
    belief_id: str
    statement: str
    timestamp: str
    retrievability: float

# Dashboard Schemas
from typing import Optional, Any

class DashboardStatsResponse(BaseModel):
    memories: int
    projects: int
    technologies: int
    careers: int
    contradictions: int

class DashboardMemoryResponse(BaseModel):
    belief_id: str
    statement: str
    timestamp: str
    event_type: str

class DashboardNameResponse(BaseModel):
    name: str
    id: Optional[str] = None
    created_at: Optional[Any] = None

class DashboardContradictionResponse(BaseModel):
    id: str
    category: str
    status: str
    severity: str
    created_at: Any

class DashboardGraphNode(BaseModel):
    id: str
    name: str
    label: str

class DashboardGraphEdge(BaseModel):
    source: str
    target: str
    type: str

class DashboardGraphResponse(BaseModel):
    nodes: list[DashboardGraphNode]
    edges: list[DashboardGraphEdge]

@router.post("/memory/store", response_model=StoreMemoryResponse, tags=["Memory Agent Operations"])
async def store_memory(payload: StoreMemoryRequest):
    """Logs raw statement, indexes semantic vector embedding, and links concept in Neo4j."""
    try:
        result = await memory_agent.store_memory(
            statement=payload.statement,
            category=payload.category
        )
        return result
    except Exception as e:
        logger.error(f"Error in store_memory API endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")

@router.get("/memory/retrieve", response_model=list[RetrieveMemoryResponse], tags=["Memory Agent Operations"])
async def retrieve_memories(query: str, limit: int = 5):
    """Hybrid search retrieving context-aware memories from Qdrant, Neo4j, and Pg."""
    try:
        results = await memory_agent.retrieve_memories(query=query, limit=limit)
        return results
    except Exception as e:
        logger.error(f"Error in retrieve_memories API endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

@router.get("/memory/decay", response_model=list[DecayLogResponse], tags=["Memory Agent Operations"])
async def get_memory_decay_logs():
    """Lists user memories and calculates dynamic retrievability values for display."""
    try:
        results = await memory_agent.get_all_memories_with_decay()
        return results
    except Exception as e:
        logger.error(f"Error in get_memory_decay_logs API endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch memory decay logs: {str(e)}")

@router.post("/conflict/resolve", response_model=ResolveContradictionResponse, tags=["Conflict Agent Operations"])
async def resolve_contradiction(payload: ResolveContradictionRequest):
    """Resolves an active contradiction state in Postgres and Neo4j."""
    try:
        await conflict_agent.resolve_contradiction(
            contradiction_id=payload.conflict_id,
            keep_belief_id=payload.keep_belief_id,
            supersede_belief_id=payload.supersede_belief_id,
            transition_trigger=payload.transition_trigger
        )
        return {
            "status": "resolved",
            "message": f"Contradiction {payload.conflict_id} resolved successfully."
        }
    except Exception as e:
        logger.error(f"Error in resolve_contradiction API endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve contradiction: {str(e)}")

@router.get("/dashboard/stats", response_model=DashboardStatsResponse, tags=["Dashboard Operations"])
async def get_dashboard_stats():
    """Fetches live count statistics from PostgreSQL and Neo4j."""
    try:
        memories_count = await postgres_db.get_memories_count()
        graph_stats = await neo4j_db.get_dashboard_stats()
        return {
            "memories": memories_count,
            "projects": graph_stats.get("projects", 0),
            "technologies": graph_stats.get("technologies", 0),
            "careers": graph_stats.get("careers", 0),
            "contradictions": graph_stats.get("contradictions", 0)
        }
    except Exception as e:
        logger.error(f"Error in get_dashboard_stats API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard stats: {str(e)}")

@router.get("/dashboard/memories", response_model=list[DashboardMemoryResponse], tags=["Dashboard Operations"])
async def get_dashboard_memories(limit: int = 50):
    """Fetches the latest memories from PostgreSQL cognitive_events table."""
    try:
        results = await postgres_db.get_latest_memories(limit=limit)
        return results
    except Exception as e:
        logger.error(f"Error in get_dashboard_memories API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard memories: {str(e)}")

@router.get("/dashboard/projects", response_model=list[DashboardNameResponse], tags=["Dashboard Operations"])
async def get_dashboard_projects():
    """Queries Neo4j for all Project node names ordered by created_at DESC."""
    try:
        results = await neo4j_db.get_all_projects()
        return results
    except Exception as e:
        logger.error(f"Error in get_dashboard_projects API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard projects: {str(e)}")

@router.get("/dashboard/technologies", response_model=list[DashboardNameResponse], tags=["Dashboard Operations"])
async def get_dashboard_technologies():
    """Queries Neo4j for all Technology node names ordered by name."""
    try:
        results = await neo4j_db.get_all_technologies()
        return results
    except Exception as e:
        logger.error(f"Error in get_dashboard_technologies API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard technologies: {str(e)}")

@router.get("/dashboard/careers", response_model=list[DashboardNameResponse], tags=["Dashboard Operations"])
async def get_dashboard_careers():
    """Queries Neo4j for all Career node names."""
    try:
        results = await neo4j_db.get_all_careers()
        return results
    except Exception as e:
        logger.error(f"Error in get_dashboard_careers API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard careers: {str(e)}")

@router.get("/dashboard/contradictions", response_model=list[DashboardContradictionResponse], tags=["Dashboard Operations"])
async def get_dashboard_contradictions():
    """Queries Neo4j for all Contradiction nodes."""
    try:
        results = await neo4j_db.get_all_contradictions()
        return results
    except Exception as e:
        logger.error(f"Error in get_dashboard_contradictions API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard contradictions: {str(e)}")

@router.get("/dashboard/graph", response_model=DashboardGraphResponse, tags=["Dashboard Operations"])
async def get_dashboard_graph():
    """Queries Neo4j for connected nodes and edges in a format suitable for graph rendering."""
    try:
        results = await neo4j_db.get_dashboard_graph()
        return results
    except Exception as e:
        logger.error(f"Error in get_dashboard_graph API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard graph: {str(e)}")

