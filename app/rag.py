"""
RAG pipeline: orchestrates the MCP server and the LangGraph agent.

This module is the main entry point for processing incoming chat messages.
It connects the MCP server (RAG tools) to the LangGraph agent via
langchain-mcp-adapters, managing conversational memory indexed by thread_id.

Execution flow:
    1. FastMCP server runs in-process via stdio mode
    2. MCP tools are converted into LangChain compatible tools
    3. The LangGraph agent is compiled with these tools
    4. The user message is dispatched to the agent
    5. The agent decides autonomously whether to call retrieve_documents
    6. The final response is sanitized and returned
"""

from contextlib import AsyncExitStack
import logging
import os
import sys
import uuid

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from app.agent import build_agent
from app.security import sanitize_output

logger = logging.getLogger(__name__)

# Cache for MCP tools and compiled agent
_client_stack: AsyncExitStack | None = None
_mcp_tools: list | None = None
_agent = None


async def init_mcp() -> list:
    """Initialize persistent stdio connection to the FastMCP server and load tools."""
    global _client_stack, _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools

    logger.info("Starting persistent MCP server connection...")

    venv_bin = os.path.dirname(sys.executable)
    fastmcp_cmd = os.path.join(venv_bin, "fastmcp.exe" if os.name == "nt" else "fastmcp")
    if not os.path.exists(fastmcp_cmd):
        fastmcp_cmd = os.path.join(venv_bin, "fastmcp")
    if not os.path.exists(fastmcp_cmd):
        fastmcp_cmd = "fastmcp"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    server_params = StdioServerParameters(
        command=fastmcp_cmd,
        args=["run", "app/mcp_server.py"],
        env=env,
    )

    _client_stack = AsyncExitStack()
    try:
        read, write = await _client_stack.enter_async_context(stdio_client(server_params))
        session = await _client_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        _mcp_tools = await load_mcp_tools(session)
        logger.info("MCP tools loaded successfully: %s", [t.name for t in _mcp_tools])
        return _mcp_tools
    except Exception:
        logger.exception("Failed to initialize MCP connection")
        await _client_stack.aclose()
        _client_stack = None
        _mcp_tools = None
        raise


async def close_mcp():
    """Cleanly close persistent connection to the MCP server."""
    global _client_stack, _mcp_tools, _agent
    if _client_stack is not None:
        logger.info("Closing persistent MCP server connection...")
        await _client_stack.aclose()
        _client_stack = None
        _mcp_tools = None
        _agent = None


def _get_agent(lang: str = "fr"):
    """Return the cached compiled agent."""
    global _agent
    if _agent is None:
        if _mcp_tools is None:
            raise RuntimeError("MCP tools not loaded. Call init_mcp() first.")
        _agent = build_agent(_mcp_tools)
    return _agent


async def get_rag_response(
    message: str,
    lang: str = "fr",
    thread_id: str | None = None,
) -> tuple[str, str]:
    """
    Process a user message through the full agentic RAG pipeline.

    Args:
        message: The sanitized user input message.
        lang: Response language preference ("fr" or "en").
        thread_id: Conversation thread identifier for memory.
                   Generates a new UUID if None.

    Returns:
        Tuple of (sanitized_reply, effective_thread_id).
    """
    effective_thread_id = thread_id or str(uuid.uuid4())
    try:
        await init_mcp()
        agent = _get_agent(lang)

        config = {"configurable": {"thread_id": effective_thread_id}}

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        reply = result["messages"][-1].content or ""
        reply = sanitize_output(reply)

        return reply, effective_thread_id

    except Exception:
        logger.exception("Error in RAG pipeline")
        if lang == "fr":
            return "Une erreur technique temporaire est survenue lors du traitement de votre demande. Veuillez réessayer dans quelques instants.", effective_thread_id
        return "A temporary technical error occurred while processing your request. Please try again in a few moments.", effective_thread_id
