"""Embeddings — singleton sentence-transformers, chargé une seule fois."""

import logging
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Charge le modèle d'embedding (appelé au démarrage de l'app)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _model


def embed_text(text: str) -> list[float]:
    """Embed un texte en vecteur. Le modèle doit être chargé au préalable."""
    model = load_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()
