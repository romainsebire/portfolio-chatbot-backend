"""
RAG pipeline — orchestre le serveur MCP et l'agent LangGraph.

Ce module est le point d'entrée principal pour le traitement des messages.
Il connecte le serveur MCP (outils RAG) à l'agent LangGraph via
langchain-mcp-adapters, et gère la mémoire conversationnelle par thread_id.

Flux complet :
    1. Le serveur MCP FastMCP est chargé en mémoire (in-process, mode stdio)
    2. Les outils MCP sont convertis en outils LangChain
    3. L'agent LangGraph est construit avec ces outils
    4. Le message utilisateur est envoyé à l'agent
    5. L'agent décide autonomement d'appeler retrieve_documents (ou pas)
    6. La réponse finale est sanitizée et renvoyée
"""

import logging
import uuid

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from app.agent import build_agent
from app.security import sanitize_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache des outils MCP et de l'agent (initialisés au premier appel)
# ---------------------------------------------------------------------------
_mcp_tools: list | None = None
_agents: dict = {}  # cache par langue : {"fr": compiled_graph, "en": compiled_graph}


async def _load_mcp_tools() -> list:
    """Charge les outils depuis le serveur MCP in-process via stdio."""
    global _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools

    logger.info("Loading MCP tools from in-process server...")

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "fastmcp", "run", "app.mcp_server:mcp"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _mcp_tools = await load_mcp_tools(session)

    logger.info("MCP tools loaded: %s", [t.name for t in _mcp_tools])
    return _mcp_tools


def _get_agent(lang: str):
    """Retourne l'agent compilé pour la langue donnée (cache)."""
    if lang not in _agents:
        # Les outils doivent être chargés avant
        if _mcp_tools is None:
            raise RuntimeError("MCP tools not loaded. Call _load_mcp_tools() first.")
        _agents[lang] = build_agent(_mcp_tools, lang=lang)
    return _agents[lang]


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------
async def get_rag_response(
    message: str,
    lang: str = "fr",
    thread_id: str | None = None,
) -> str:
    """
    Traite un message utilisateur via le pipeline agentic RAG complet.

    Args:
        message: Le message sanitizé de l'utilisateur.
        lang: Langue de la réponse ("fr" ou "en").
        thread_id: Identifiant de conversation pour la mémoire.
                   Si None, génère un UUID (conversation one-shot).

    Returns:
        La réponse textuelle de l'agent, sanitizée.
    """
    try:
        # 1. Charger les outils MCP (lazy, une seule fois)
        await _load_mcp_tools()

        # 2. Récupérer l'agent compilé
        agent = _get_agent(lang)

        # 3. Préparer le thread_id pour la mémoire
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        # 4. Invoquer l'agent avec le message
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        # 5. Extraire la dernière réponse de l'agent
        reply = result["messages"][-1].content or ""

        # 6. Sanitize la sortie (protection XSS)
        reply = sanitize_output(reply)

        return reply

    except Exception:
        logger.exception("Error in RAG pipeline")
        if lang == "fr":
            return "Désolé, une erreur est survenue. Veuillez réessayer dans quelques instants."
        return "Sorry, an error occurred. Please try again in a moment."
