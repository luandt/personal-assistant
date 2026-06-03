"""
Build and compile the LangGraph agent graph with PostgreSQL checkpointing.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.state import AgentState
from app.agent.nodes import make_nodes, should_continue
from app.agent.tools import make_todo_tools
from app.config import get_settings

settings = get_settings()

_compiled_graph = None


async def build_graph(db_session_factory):
    """
    Build and compile the agent graph.
    Call once at startup; reuse the returned graph for all requests.
    """
    global _compiled_graph

    # Create tools bound to the DB session factory
    tools = make_todo_tools(db_session_factory)

    # Create node functions
    call_llm, execute_tools = make_nodes(tools)

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("call_llm", call_llm)
    builder.add_node("execute_tools", execute_tools)

    builder.set_entry_point("call_llm")

    builder.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )

    # After executing tools, loop back to LLM to process results
    builder.add_edge("execute_tools", "call_llm")

    # Set up PostgreSQL checkpointer for conversation memory
    try:
        checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url_sync)
        # AsyncPostgresSaver handles initialization internally when needed
        _compiled_graph = builder.compile(checkpointer=checkpointer)
    except Exception as e:
        print(f"Warning: Checkpointer initialization failed: {e}")
        print("Compiling graph without checkpointer (no conversation history)")
        _compiled_graph = builder.compile()
    
    return _compiled_graph


async def get_graph():
    """Return the compiled graph (must call build_graph first)."""
    if _compiled_graph is None:
        raise RuntimeError("Graph not built. Call build_graph() at startup.")
    return _compiled_graph


async def run_agent(graph, user_id: str, telegram_id: str, chat_id: str, user_message: str) -> str:
    """
    Invoke the agent for a user message and return the assistant's text response.
    thread_id is per-user to maintain conversation history.
    """
    from langchain_core.messages import HumanMessage

    config = {
        "configurable": {
            "thread_id": f"user_{telegram_id}",
        }
    }

    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "telegram_id": telegram_id,
        "chat_id": chat_id,
        "intent": "",
        "entities": {},
        "current_todos": [],
        "last_action": "",
        "response_to_user": "",
    }

    result = await graph.ainvoke(input_state, config=config)

    # Extract the last AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            content = msg.content
            if isinstance(content, list):
                # Handle structured content blocks
                text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
                return " ".join(text_parts).strip() or "Done!"
            return str(content).strip() or "Done!"

    return "I processed your request."
