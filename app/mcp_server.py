"""
MCP RAG Server: FastMCP server exposing vector search tools.

This server runs in-process (stdio mode) alongside the agent.
It exposes two MCP tools that the LangGraph agent can call autonomously:
  - retrieve_documents: semantic search in Pinecone
  - get_context_info: metadata regarding the vector database index
"""

import logging

from fastmcp import FastMCP
from pinecone import Pinecone

from app.config import settings
from app.embeddings import embed_text

logger = logging.getLogger(__name__)

# FastMCP server instance (consumed via stdio by the agent)
mcp = FastMCP("Portfolio RAG")

# Pinecone client (lazy singleton)
_pc_index = None


def _get_pinecone_index():
    global _pc_index
    if _pc_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pc_index


# MCP Tools
TOP_K = 3


@mcp.tool()
def retrieve_documents(query: str, top_k: int = TOP_K) -> dict:
    """
    Semantic search in Romain Sebire's portfolio vector database.

    Use this function to answer questions about Romain's background, profile,
    skills, technical projects, and work experiences.

    Args:
        query: Search query or question in natural language (English keywords).
        top_k: Number of results to return (1 to 3, default: 3).
               - Use top_k=1 for a specific sub-section (e.g. RESULTS, METHODOLOGY, CONTEXT).
               - Use top_k=3 for a broad overview of a project or full summary.

    Returns:
        Dictionary containing matching document chunks with their source section and similarity score.
    """
    try:
        actual_top_k = min(top_k, TOP_K)

        # 1. Embed query
        query_embedding = embed_text(query)

        # 2. Query Pinecone vector database
        index = _get_pinecone_index()
        results = index.query(
            vector=query_embedding,
            top_k=actual_top_k,
            include_metadata=True,
        )

        # 3. Format and filter results using minimum similarity score
        documents = []
        min_score = getattr(settings, "MIN_SIMILARITY_SCORE", 0.75)
        for match in results.get("matches", []):
            score = round(match.get("score", 0), 4)
            if score < min_score:
                logger.info(f"Filtered out document with score {score} < {min_score}")
                continue
            metadata = match.get("metadata", {})
            documents.append({
                "text": metadata.get("text", ""),
                "section": metadata.get("section", ""),
                "score": score,
            })

        logger.info(
            f"[TOOL CALL] retrieve_documents(query='{query[:40]}...', top_k={top_k}) -> Found {len(documents)} documents"
        )

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
    Return statistics and information about the portfolio vector index.

    Use this function to check vector database status or inspect available metadata.

    Returns:
        Dictionary containing Pinecone index statistics.
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
