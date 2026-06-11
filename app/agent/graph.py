"""
Build and compile the LangGraph agent graph with PostgreSQL checkpointing.
"""
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import make_nodes, should_continue, should_continue_llm
from app.agent.tools import make_todo_tools
from app.config import get_settings
import psycopg
import logging
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

_compiled_graph = None


async def build_graph(db_session_factory, store=None):
    """
    Build and compile the agent graph.
    Call once at startup; reuse the returned graph for all requests.
    """
    global _compiled_graph

    # Create tools bound to the DB session factory
    tools = make_todo_tools(db_session_factory)

    # Create node functions
    classify_intent, ask_clarification, todo_llm, execute_tools = make_nodes(tools)
 
    builder = StateGraph(AgentState)
    
    # Nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("todo_llm", todo_llm)
    builder.add_node("execute_tools", execute_tools)
 
    # Entry: always classify first
    builder.set_entry_point("classify_intent")
 
    # Route 1: after classification
    # - If low confidence → ask for clarification → END
    # - Else → call LLM (with intent info injected)
    builder.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "ask_clarification": "ask_clarification",
            "todo_llm": "todo_llm",
        },
    )
 
    # After clarification, end the turn (wait for user response)
    builder.add_edge("ask_clarification", END)
 
    # Route 2: after LLM call
    # - If has tool_calls → execute
    # - Else → end
    builder.add_conditional_edges(
        "todo_llm",
        should_continue_llm,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )

    builder.add_edge("execute_tools", "todo_llm")


    # Set up PostgreSQL checkpointer for conversation memory
    try:
        # checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url_sync)
        conn_string = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")

        conn = await psycopg.AsyncConnection.connect(
            conn_string,
            autocommit=True,  # required for AsyncPostgresSaver
        )
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()  # creates checkpoint tables in postgres

        # AsyncPostgresSaver handles initialization internally when needed
        compile_kwargs = {"checkpointer": checkpointer}
        if store is not None:
            compile_kwargs["store"] = store
        _compiled_graph = builder.compile(**compile_kwargs)
        logger.info(f"✅ Graph compiled with checkpointer")

    except Exception as e:
        print(f"Warning: Checkpointer initialization failed: {e}")
        print("Compiling graph without checkpointer (no conversation history)")
        # If a custom store was provided, try compiling with it even if checkpointer failed
        if store is not None:
            _compiled_graph = builder.compile(store=store)
        else:
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
        },
        
        # "configurable": {"thread_id": thread_id},
        "metadata": {
            "user_id": user_id,
            "telegram_chat_id": chat_id,
        },
        "tags": ["telegram", "todo-bot"],
    }

    # checkpoint = await graph.aget_state(config)
    # logger.info(f"Existing messages in thread: {len(checkpoint.values.get('messages', []))}")

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
        "confidence": 0.0,
        "ambiguous_fields": [],
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
