"""Embedding generation using the Pinecone Inference API."""

import logging
from pinecone import Pinecone
from app.config import settings

logger = logging.getLogger(__name__)

_pc_client = None


def _get_pinecone_client() -> Pinecone:
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc_client


def embed_text(text: str, input_type: str = "query") -> list[float]:
    """
    Generate an embedding vector for a given text via Pinecone Inference API.

    Args:
        text: The text string to vectorise.
        input_type: "query" for retrieval, "passage" for indexing a document.
    """
    client = _get_pinecone_client()
    try:
        response = client.inference.embed(
            model=settings.EMBEDDING_MODEL,
            inputs=[text],
            parameters={
                "input_type": input_type,
                "truncate": "END",
            },
        )
        return response.data[0].values
    except Exception as e:
        logger.error("Failed to generate embedding via Pinecone: %s", e)
        raise
