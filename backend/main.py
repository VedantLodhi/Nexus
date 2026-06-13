import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

# Database connectors
from app.db.postgres import postgres_db
from app.db.neo4j import neo4j_db
from app.db.qdrant import qdrant_db
from app.config import settings

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nexus.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle handler wrapping connection start and graceful shutdown."""
    logger.info("Initializing NexusOS MVP database integrations...")
    try:
        # Asynchronously connect databases
        await postgres_db.connect()
        await neo4j_db.connect()
        await qdrant_db.connect()
        logger.info("All database integrations successfully connected.")
        yield
    except Exception as e:
        logger.critical(f"Database initialization failed during app startup: {e}")
        raise e
    finally:
        logger.info("Cleaning up NexusOS databases on shutdown...")
        await postgres_db.disconnect()
        await neo4j_db.disconnect()
        await qdrant_db.disconnect()
        logger.info("Graceful shutdown cleanup complete.")

# Initialize app
app = FastAPI(
    title="NexusOS MVP Backend API",
    description="Cognitive Digital Twin Operating System Core Infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routing setup
api_router = APIRouter(prefix="/api/v1")

from app.api.endpoints import router as memory_router
api_router.include_router(memory_router)


@api_router.get("/health", tags=["System Diagnostics"])
async def health_check():
    """Performs deep query checks across Pg, Neo4j, and Qdrant to verify connectivity."""
    status_payload = {
        "status": "healthy",
        "timestamp": time.time(),
        "databases": {
            "postgres": "unknown",
            "neo4j": "unknown",
            "qdrant": "unknown"
        }
    }
    
    # 1. Check PostgreSQL
    try:
        if postgres_db.pool:
            async with postgres_db.pool.acquire() as conn:
                res = await conn.fetchval("SELECT 1")
                if res == 1:
                    status_payload["databases"]["postgres"] = "healthy"
                else:
                    status_payload["databases"]["postgres"] = "degraded"
        else:
            status_payload["databases"]["postgres"] = "uninitialized"
    except Exception as e:
        status_payload["databases"]["postgres"] = f"unhealthy: {str(e)}"
        status_payload["status"] = "degraded"

    # 2. Check Neo4j
    try:
        if neo4j_db.driver:
            await neo4j_db.driver.verify_connectivity()
            status_payload["databases"]["neo4j"] = "healthy"
        else:
            status_payload["databases"]["neo4j"] = "uninitialized"
    except Exception as e:
        status_payload["databases"]["neo4j"] = f"unhealthy: {str(e)}"
        status_payload["status"] = "degraded"

    # 3. Check Qdrant
    try:
        if qdrant_db.client:
            await qdrant_db.client.get_collections()
            status_payload["databases"]["qdrant"] = "healthy"
        else:
            status_payload["databases"]["qdrant"] = "uninitialized"
    except Exception as e:
        status_payload["databases"]["qdrant"] = f"unhealthy: {str(e)}"
        status_payload["status"] = "degraded"

    return status_payload

# Register routers
app.include_router(api_router)
