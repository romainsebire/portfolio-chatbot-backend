"""
Agent LangGraph — graphe ReAct avec tool-calling et mémoire conversationnelle.

Architecture :
    START → [call_model] → (tools_condition) →
        si tool calls : [tools] → retour à [call_model]
        sinon         : END

L'agent utilise ChatGroq comme LLM, avec les outils MCP bindés.
La mémoire est gérée par MemorySaver (par thread_id pour le multi-turn).
"""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.prompts import SYSTEM_PROMPT_FR, SYSTEM_PROMPT_EN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State — état du graphe (liste de messages avec accumulation automatique)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# Mémoire conversationnelle (in-process, par thread_id)
# ---------------------------------------------------------------------------
memory = MemorySaver()


# ---------------------------------------------------------------------------
# Construction du graphe
# ---------------------------------------------------------------------------
def build_agent(tools: list, lang: str = "fr"):
    """
    Construit et compile le graphe LangGraph ReAct.

    Args:
        tools: Liste des outils MCP à binder au modèle.
        lang: Langue pour le system prompt ("fr" ou "en").

    Returns:
        Le graphe compilé avec la mémoire, prêt à être invoqué.
    """
    # 1. Initialiser le LLM avec les outils bindés
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=0.3,
        max_tokens=500,
        api_key=settings.GROQ_API_KEY,
    )
    llm_with_tools = llm.bind_tools(tools)

    # 2. Choisir le system prompt selon la langue
    system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN

    # 3. Définir le nœud "call_model"
    def call_model(state: AgentState):
        """Nœud principal : appelle le LLM avec le system prompt et l'historique."""
        messages = state["messages"]

        # Injecter le system prompt en première position s'il n'y est pas
        from langchain_core.messages import SystemMessage
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 4. Construire le graphe
    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    # 5. Compiler avec la mémoire
    return builder.compile(checkpointer=memory)
