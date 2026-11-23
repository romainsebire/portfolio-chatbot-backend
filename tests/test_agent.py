"""Regression tests for multi-turn agent state preparation, tool binding, and language fallback."""

import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.agent import _prepare_model_inputs, build_agent
from app.prompts import SYSTEM_PROMPT_PASS1, SYSTEM_PROMPT_PASS2


def test_prepare_model_inputs_resets_pass1_on_new_turn_after_past_tool_message():
    """A new HumanMessage must start in Pass 1 even when earlier turns used tools."""
    messages = [
        HumanMessage(content="Quelles sont les thématiques traitées par les projets de Romain ?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_documents", "args": {"query": "SUMMARY"}, "id": "call_1"}],
        ),
        ToolMessage(content="SUMMARY : PROJECTS BY TECHNICAL DOMAIN...", tool_call_id="call_1"),
        AIMessage(content="Les projets couvrent l'IA, le Machine Learning et l'ingénierie logicielle."),
        HumanMessage(content="Donne moi un projet majeur"),
    ]

    is_pass2, prepared = _prepare_model_inputs(messages, lang="fr")

    assert is_pass2 is False
    assert isinstance(prepared[0], SystemMessage)
    assert SYSTEM_PROMPT_PASS1 in prepared[0].content
    assert [m.content for m in prepared[1:]] == [
        "Quelles sont les thématiques traitées par les projets de Romain ?",
        "Les projets couvrent l'IA, le Machine Learning et l'ingénierie logicielle.",
        "Donne moi un projet majeur",
    ]
    assert not any(isinstance(m, ToolMessage) for m in prepared)


def test_prepare_model_inputs_enters_pass2_only_on_current_turn_tool_message():
    """Pass 2 activates when a ToolMessage is emitted within the current turn."""
    messages = [
        HumanMessage(content="Tour 1"),
        ToolMessage(content="Doc tour 1", tool_call_id="call_1"),
        AIMessage(content="Réponse tour 1"),
        HumanMessage(content="Donne moi un projet majeur"),
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_documents", "args": {"query": "PROJECT"}, "id": "call_2"}],
        ),
        ToolMessage(content="PROJECT : Cybersecurity Dataset Construction...", tool_call_id="call_2"),
    ]

    is_pass2, prepared = _prepare_model_inputs(messages, lang="fr")

    assert is_pass2 is True
    assert isinstance(prepared[0], SystemMessage)
    assert SYSTEM_PROMPT_PASS2 in prepared[0].content
    tool_msgs = [m for m in prepared if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_2"


def test_prepare_model_inputs_sliding_window_keeps_complete_turns():
    """History pruning keeps up to 3 complete turns (6 messages), never starting with an AIMessage."""
    messages = []
    for i in range(1, 6):
        messages.append(HumanMessage(content=f"Q{i}"))
        messages.append(
            AIMessage(
                content="Thinking preamble",
                tool_calls=[{"name": "retrieve_documents", "args": {}, "id": f"call_{i}"}],
            )
        )
        messages.append(ToolMessage(content=f"Doc{i}", tool_call_id=f"call_{i}"))
        messages.append(AIMessage(content=f"A{i}"))
    messages.append(HumanMessage(content="Q6"))

    is_pass2, prepared = _prepare_model_inputs(messages, lang="fr")

    assert is_pass2 is False
    history_plus_current = prepared[1:]
    assert [m.content for m in history_plus_current] == [
        "Q3", "A3", "Q4", "A4", "Q5", "A5", "Q6",
    ]
    assert isinstance(history_plus_current[0], HumanMessage)


def test_prepare_model_inputs_language_fallback_hint():
    """Interface language fallback reflects the active portfolio page language."""
    _, prepared_fr = _prepare_model_inputs([HumanMessage(content="Sakut")], lang="fr")
    _, prepared_en = _prepare_model_inputs([HumanMessage(content="Sakut")], lang="en")

    assert "French" in prepared_fr[0].content
    assert "English" in prepared_en[0].content


def test_multiturn_graph_rebinds_tools_on_second_turn():
    """Verify compiled LangGraph agent keeps binding tools on Turn 2+ and switches lang dynamically."""
    executed_queries = []

    @tool
    def dummy_retrieve(query: str) -> str:
        """Mock RAG retrieval tool."""
        executed_queries.append(query)
        return f"CONTEXT for {query}: Cybersecurity SAST LLM dataset with 12,000 samples."

    call_log = []

    class FakeBoundLLM:
        def invoke(self, messages):
            sys_content = messages[0].content
            turn_num = len(executed_queries) + 1
            call_log.append(("bound", sys_content))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dummy_retrieve",
                        "args": {"query": f"query_{turn_num}"},
                        "id": f"call_{turn_num}",
                    }
                ],
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    class FakeChatGroq:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            return FakeBoundLLM()

        def invoke(self, messages):
            sys_content = messages[0].content
            call_log.append(("unbound", sys_content))
            assert any(isinstance(m, ToolMessage) for m in messages)
            return AIMessage(
                content=f"Grounded answer #{len(executed_queries)}",
                usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
            )

    with (
        patch("app.agent.ChatGroq", FakeChatGroq),
        patch("app.security.token_limiter.add_tokens"),
    ):
        agent = build_agent([dummy_retrieve])
        thread_id = f"test-multiturn-{uuid.uuid4().hex}"

        out1 = agent.invoke(
            {"messages": [HumanMessage(content="Quelles sont les thématiques ?")]},
            config={"configurable": {"thread_id": thread_id, "lang": "fr"}},
        )
        out2 = agent.invoke(
            {"messages": [HumanMessage(content="Donne moi un projet majeur")]},
            config={"configurable": {"thread_id": thread_id, "lang": "en"}},
        )

    assert out1["messages"][-1].content == "Grounded answer #1"
    assert out2["messages"][-1].content == "Grounded answer #2"
    assert executed_queries == ["query_1", "query_2"]

    # Call sequence across 2 turns must be: Pass 1 (bound), Pass 2 (unbound), Pass 1 (bound), Pass 2 (unbound)
    assert [mode for mode, _ in call_log] == ["bound", "unbound", "bound", "unbound"]
    assert SYSTEM_PROMPT_PASS1 in call_log[0][1]
    assert "French" in call_log[0][1]
    assert SYSTEM_PROMPT_PASS2 in call_log[1][1]
    assert SYSTEM_PROMPT_PASS1 in call_log[2][1]
    assert "English" in call_log[2][1]
    assert SYSTEM_PROMPT_PASS2 in call_log[3][1]
