import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.engine.memory import memory_agent

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
        from app.engine.conflict import conflict_agent
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
