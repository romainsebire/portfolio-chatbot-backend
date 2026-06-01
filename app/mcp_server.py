"""
MCP Server RAG — serveur FastMCP exposant les outils de recherche vectorielle.

Ce serveur tourne IN-PROCESS (mode stdio) dans le même processus que l'agent.
Il expose deux tools MCP que l'agent LangGraph peut appeler autonomement :
  - retrieve_documents : recherche sémantique dans Pinecone
  - get_context_info   : méta-informations sur la base vectorielle
"""

import logging

from fastmcp import FastMCP
from pinecone import Pinecone

from app.config import settings
from app.embeddings import embed_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server (sera consommé en mode stdio par l'agent)
# ---------------------------------------------------------------------------
mcp = FastMCP("Portfolio RAG")

# ---------------------------------------------------------------------------
# Pinecone client (lazy singleton)
# ---------------------------------------------------------------------------
_pc_index = None


def _get_pinecone_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pc_index


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
TOP_K = 5


@mcp.tool()
def retrieve_documents(query: str, top_k: int = TOP_K) -> dict:
    """
    Recherche sémantique dans la base vectorielle du portfolio de Romain Sebire.

    Utilise cette fonction pour répondre aux questions sur le profil, le parcours,
    les compétences, les projets et les expériences de Romain.

    Args:
        query: La question ou requête de recherche en langage naturel.
        top_k: Nombre de résultats à retourner (défaut: 5).

    Returns:
        Dictionnaire contenant les documents pertinents trouvés avec leur
        section d'origine et leur score de similarité.
    """
    try:
        # 1. Embed la requête
        query_embedding = embed_text(query)

        # 2. Recherche Pinecone
        index = _get_pinecone_index()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        # 3. Formatter les résultats
        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            documents.append({
                "text": metadata.get("text", ""),
                "section": metadata.get("section", ""),
                "score": round(match.get("score", 0), 4),
            })

        return {
            "success": True,
            "query": query,
            "documents_found": len(documents),
            "documents": documents,
        }

    except Exception as e:
        logger.exception("Error in retrieve_documents")
        return {
            "success": False,
            "error": str(e),
            "documents": [],
        }


@mcp.tool()
def get_context_info() -> dict:
    """
    Retourne des informations sur la base vectorielle du portfolio.

    Utilise cette fonction pour vérifier l'état de la base de données
    ou savoir quelles informations sont disponibles.

    Returns:
        Dictionnaire avec les statistiques de l'index Pinecone.
    """
    try:
        index = _get_pinecone_index()
        stats = index.describe_index_stats()
        return {
            "success": True,
            "index_name": settings.PINECONE_INDEX_NAME,
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
        }
    except Exception as e:
        logger.exception("Error in get_context_info")
        return {"success": False, "error": str(e)}
