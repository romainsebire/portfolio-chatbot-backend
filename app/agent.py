"""
LangGraph agent: ReAct graph with tool calling and conversational memory.

Architecture:
    START -> [call_model] -> (tools_condition) ->
        if tool calls: [tools] -> back to [call_model]
        else         : END

The agent uses ChatGroq as the LLM provider with bound MCP tools.
Memory is managed by MemorySaver indexed by thread_id for multi-turn conversations.
"""

import logging
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.prompts import SYSTEM_PROMPT_PASS1, SYSTEM_PROMPT_PASS2

logger = logging.getLogger(__name__)


# Graph state (message list with automatic append strategy)
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Conversational memory (in-process, indexed by thread_id)
memory = MemorySaver()


class ModelCooldownTracker:
    """In-memory tracker for temporary quarantine of Groq models encountering rate limits."""

    def __init__(self):
        self._disabled_until: dict[str, float] = {}

    def is_active(self, model_name: str) -> bool:
        """Check whether a model is available (quarantine expired or inactive)."""
        until = self._disabled_until.get(model_name, 0)
        return time.time() >= until

    def parse_and_disable(self, model_name: str, error_message: str):
        """Extract cooldown duration from Groq error message and quarantine the model."""
        msg = str(error_message)
        match = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s', msg, re.IGNORECASE)
        if match:
            h = int(match.group(1) or 0)
            mins = int(match.group(2) or 0)
            secs = float(match.group(3) or 0)
            cooldown = h * 3600 + mins * 60 + secs
        else:
            cooldown = 60.0

        cooldown = max(cooldown, 10.0)
        self._disabled_until[model_name] = time.time() + cooldown
        logger.warning(
            f"[CIRCUIT BREAKER] Model '{model_name}' placed in quarantine for {cooldown:.1f}s. "
            f"Subsequent requests will skip this model."
        )


model_tracker = ModelCooldownTracker()


def _build_system_prompt(is_pass2: bool, lang: str = "fr") -> str:
    """Return the Pass 1 or Pass 2 system prompt with interface language fallback instructions."""
    base_prompt = SYSTEM_PROMPT_PASS2 if is_pass2 else SYSTEM_PROMPT_PASS1
    default_lang = "English" if lang == "en" else "French"
    return (
        f"{base_prompt}\n"
        "## INTERFACE LANGUAGE FALLBACK\n"
        f"- The visitor is browsing the {default_lang} version of the portfolio. Always reply in the explicit language of the visitor's message. "
        f"If the message is an ambiguous short greeting, single word, or typo (e.g., 'Sakut', 'Hey', 'Ok'), default to {default_lang}.\n"
    )


def _prepare_model_inputs(
    messages: list[AnyMessage],
    lang: str = "fr",
) -> tuple[bool, list[AnyMessage]]:
    """
    Determine whether the current turn is in Pass 2 and construct bounded inputs for the LLM.

    Pass 2 is active strictly when a ToolMessage exists in the current turn (at or after the last
    HumanMessage), preventing past-turn tool calls in MemorySaver from disabling tool binding on
    subsequent turns.
    """
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if human_indices:
        last_human_idx = human_indices[-1]
        current_turn_messages = list(messages[last_human_idx:])
        is_pass2 = any(isinstance(m, ToolMessage) for m in current_turn_messages)
        active_system_prompt = _build_system_prompt(is_pass2=is_pass2, lang=lang)

        pruned_past: list[AnyMessage] = []
        for msg in messages[:last_human_idx]:
            if isinstance(msg, HumanMessage):
                pruned_past.append(msg)
            elif (
                isinstance(msg, AIMessage)
                and not getattr(msg, "tool_calls", None)
                and msg.content
                and str(msg.content).strip()
            ):
                pruned_past.append(AIMessage(content=str(msg.content).strip()))

        # Sliding window: retain up to 3 complete past turns (6 messages) without an orphan leading AIMessage
        pruned_past = pruned_past[-6:]
        while pruned_past and isinstance(pruned_past[0], AIMessage):
            pruned_past.pop(0)

        prepared_messages = [SystemMessage(content=active_system_prompt)] + pruned_past + current_turn_messages
        return is_pass2, prepared_messages

    is_pass2 = any(isinstance(m, ToolMessage) for m in messages)
    active_system_prompt = _build_system_prompt(is_pass2=is_pass2, lang=lang)
    if not messages or not isinstance(messages[0], SystemMessage):
        return is_pass2, [SystemMessage(content=active_system_prompt)] + list(messages)
    return is_pass2, list(messages)


def build_agent(tools: list, lang: str = "fr"):
    """
    Build and compile the LangGraph ReAct execution graph.

    Args:
        tools: List of MCP tools to bind to the model.
        lang: Language preference string.

    Returns:
        Compiled execution graph with checkpointer.
    """
    def call_model(state: AgentState, config: RunnableConfig | None = None):
        """Main graph node: invokes LLM using dynamic system prompt (Pass 1 or Pass 2) and pruned history."""
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        effective_lang = configurable.get("lang") or lang
        is_pass2, messages = _prepare_model_inputs(state["messages"], lang=effective_lang)

        # Ordered fallback cascade over Groq models supporting tool calling
        base_cascade = settings.groq_cascade_list

        cascade = []
        for model in base_cascade:
            if model and model not in cascade:
                cascade.append(model)

        # Filter out quarantined models via circuit breaker
        active_cascade = [m for m in cascade if model_tracker.is_active(m)]
        if not active_cascade:
            active_cascade = cascade  # Fallback to full cascade if all are paused

        response = None
        last_error = None

        for model_name in active_cascade:
            try:
                current_llm = ChatGroq(
                    model=model_name,
                    temperature=0.0,
                    max_tokens=1024,
                    api_key=settings.GROQ_API_KEY,
                    max_retries=1,
                    timeout=20.0,
                )
                # Only bind MCP tools in Pass 1 (Router). In Pass 2 (Generator), invoke pure LLM
                # so the model cannot loop on tool calls and saves tool schema prompt tokens.
                llm_to_invoke = current_llm if is_pass2 else current_llm.bind_tools(tools)
                response = llm_to_invoke.invoke(messages)

                # Strip reasoning tags (<think> and #thought)
                if isinstance(response.content, str):
                    response.content = re.sub(r"(?:<think>|#thought).*?(?:</think>|\n\n(?=[A-Z\*\#\-]))", "", response.content, flags=re.DOTALL).strip()
                    response.content = re.sub(r"^#thought.*?\n\n", "", response.content, flags=re.DOTALL).strip()

                # Response validation: empty text with no tool calls triggers fallback
                if not response.content.strip() and not getattr(response, "tool_calls", None):
                    raise ValueError("Model returned an empty response without tool calls.")

                usage = getattr(response, "usage_metadata", {}) or {}
                prompt_tok = usage.get("input_tokens", 0)
                completion_tok = usage.get("output_tokens", 0)
                total_tok = usage.get("total_tokens", 0)

                from app.security import token_limiter, current_client_ip
                client_ip = current_client_ip.get()
                token_limiter.add_tokens(client_ip, total_tok)

                logger.info(
                    f"[LLM TOKEN USAGE] IP: {client_ip} | Model: {model_name} | "
                    f"Prompt: {prompt_tok} | Completion: {completion_tok} | Total: {total_tok}"
                )

                break
            except Exception as e:
                last_error = e
                if "rate limit" in str(e).lower() or "429" in str(e):
                    model_tracker.parse_and_disable(model_name, str(e))
                else:
                    logger.warning(
                        f"Failure with model '{model_name}' in cascade ({e}). "
                        f"Falling back to next model..."
                    )
        else:
            if last_error:
                raise last_error

        return {"messages": [response]}

    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=memory)
