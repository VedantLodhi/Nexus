import logging
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from app.config import settings

logger = logging.getLogger("nexus.qdrant")

class QdrantManager:
    def __init__(self):
        self.client: AsyncQdrantClient | None = None

    async def connect(self):
        """Initialize Async Qdrant Client and setup the memory collections."""
        if self.client is not None:
            return

        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
        try:
            self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
            # Basic health check ping
            await self.client.get_collections()
            logger.info("Qdrant database connection verified.")
            await self.initialize_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise e

    async def disconnect(self):
        """Teardown Qdrant Connection Client."""
        if self.client is None:
            return

        logger.info("Closing Qdrant connection client...")
        await self.client.close()
        self.client = None
        logger.info("Qdrant connection client closed.")

    async def initialize_collections(self):
        """Ensures the semantic_memory collection exists with a 768-dim index."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized.")

        collection_name = "semantic_memory"
        
        try:
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                logger.info(f"Collection '{collection_name}' not found. Creating collection...")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # Cosine matching size for nomic-embed-text
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collections: {e}")
            raise e

    async def upsert_memory_vector(self, belief_id: str, vector: list[float], statement: str, user_id: str):
        """Inserts or updates a vector point in the Qdrant semantic_memory collection."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized.")

        point = models.PointStruct(
            id=belief_id,
            vector=vector,
            payload={
                "belief_id": belief_id,
                "user_id": user_id,
                "statement": statement,
                "confidence": 1.0,
                "last_reinforced": int(time.time())
            }
        )
        
        await self.client.upsert(
            collection_name="semantic_memory",
            points=[point]
        )
        logger.debug(f"Upserted vector point in Qdrant for belief_id: {belief_id}")

    async def search_similar_memories(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Performs cosine similarity search against Qdrant semantic_memory."""
        if self.client is None:
            raise RuntimeError("Qdrant client not initialized.")

        results = await self.client.search(
            collection_name="semantic_memory",
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "belief_id": hit.payload.get("belief_id"),
                "statement": hit.payload.get("statement"),
                "similarity": hit.score,
                "user_id": hit.payload.get("user_id"),
                "confidence": hit.payload.get("confidence", 1.0)
            }
            for hit in results
        ]

    async def delete_memory_vector(self, belief_id: str):
        """Deletes a vector point from Qdrant by ID (used for transaction rollbacks)."""
        if self.client is None:
            return
        
        await self.client.delete(
            collection_name="semantic_memory",
            points_selector=models.PointIdsList(
                points=[belief_id]
            )
        )
        logger.info(f"Qdrant vector point '{belief_id}' deleted for rollback.")

# Import time for payload epoch timestamp
import time

# Singleton instance
qdrant_db = QdrantManager()


