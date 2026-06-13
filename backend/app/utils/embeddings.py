import logging
import httpx
from app.config import settings

logger = logging.getLogger("nexus.embeddings")

async def get_ollama_embedding(text: str) -> list[float]:
    """Calls local Ollama API to generate a dense vector embedding using nomic-embed-text."""
    url = f"{settings.OLLAMA_URL}/api/embeddings"
    payload = {
        "model": settings.OLLAMA_EMBED_MODEL,
        "prompt": text
    }
    
    logger.debug(f"Requesting embedding from Ollama for text: {text[:30]}...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not embedding:
                raise ValueError("Ollama response did not contain an embedding payload.")
            return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding from Ollama: {e}")
        raise e
