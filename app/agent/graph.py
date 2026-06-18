"""
Build and compile the LangGraph agent graph with PostgreSQL checkpointing.
"""
import uuid

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    make_nodes,
    messages_router,
    should_execute_tools,
    should_search_web_after_chat,
    should_continue_after_calendar,
)
from app.agent.tools import make_todo_tools
from app.config import get_settings
import psycopg
import logging
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

CALENDAR_READ_ONLY_TOOLS = {
    "list-calendars",
    "list-events",
    "search-events",
    "get-event",
    "list-colors",
    "get-freebusy",
    "get-current-time",
}

_compiled_graph = None


async def build_graph(db_session_factory, store=None):
    """
    Build and compile the agent graph.
    Call once at startup; reuse the returned graph for all requests.
    """
    global _compiled_graph

    # Create tools bound to the DB session factory
    # tools = make_todo_tools(db_session_factory)

    # ── 1. Todo tools (existing) ───────────────────────────────
    todo_tools = make_todo_tools(db_session_factory)

    # ── 2. Google Calendar MCP tools ───────────────────────────
    calendar_tools = []
    mcp_client = None

    print("Attempting to load Google Calendar MCP tools...")

    try:
        mcp_client = MultiServerMCPClient({
            "google-calendar": {
                "command": "npx",
                "args": ["-y", "@cocal/google-calendar-mcp"],
                "transport": "stdio",
                "env": {
                    "GOOGLE_OAUTH_CREDENTIALS": settings.google_credentials_file,
                },
            }
        })
        calendar_tools = await mcp_client.get_tools()
        calendar_tools = [
            tool for tool in calendar_tools
            if tool.name in CALENDAR_READ_ONLY_TOOLS
        ]
        logger.info(
            f"✅ Google Calendar MCP read-only tools loaded: {[t.name for t in calendar_tools]}"
        )

    except Exception as e:
        logger.warning(
            f"⚠️  Google Calendar MCP unavailable: {e}\n"
            f"   Conflict checking will be skipped."
        )
    
    all_tools = todo_tools + calendar_tools

    # Create node functions
    
    classify_intent, ask_clarification, todo_llm, execute_tools, web_search_node, chat_response_node, check_calendar_node, ask_calendar_confirmation = make_nodes(all_tools)
 
    builder = StateGraph(AgentState)
    
    # Nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("check_calendar", check_calendar_node)
    builder.add_node("ask_calendar_confirmation", ask_calendar_confirmation)
    builder.add_node("todo_llm", todo_llm)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("web_search", web_search_node)
    builder.add_node("chat_response_node", chat_response_node)
 
    # Entry: always classify first
    builder.set_entry_point("classify_intent")
 
    # Route 1: after classification
    builder.add_conditional_edges(
        "classify_intent",
        messages_router,
        {
            "chat_response_node": "chat_response_node",
            "web_search": "web_search",
            "ask_clarification": "ask_clarification",
            "todo_llm": "todo_llm",
            "check_calendar": "check_calendar",
        },
    )

    # Route 2: check_calendar → conflict? ask confirmation : todo_llm
    builder.add_conditional_edges(
        "check_calendar",
        should_continue_after_calendar,
        {
            "ask_calendar_confirmation": "ask_calendar_confirmation",
            "todo_llm": "todo_llm",
        },
    )
 
    # After clarification, end the turn (wait for user response)
    builder.add_edge("ask_clarification", END)

    # After web search, end the turn
    builder.add_edge("web_search", END)
    # builder.add_edge("chat_response_node", END)
    builder.add_conditional_edges(
        "chat_response_node",
        should_search_web_after_chat,
        {
            "web_search": "web_search",
            "end": END,
        },
    )
 
    builder.add_conditional_edges(
        "todo_llm",
        should_execute_tools,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )

    builder.add_edge("execute_tools", "todo_llm")
    logger.info(f"✅ Nodes and edges added to builder")
    logger.info(f"DEBUG nodes in builder: {list(builder.nodes.keys())}")

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

    print(f"Running agent for user_id={user_id}, telegram_id={telegram_id}, chat_id={chat_id}, message='{user_message}'")

    config = {
        "configurable": {
            "thread_id": f"user_{telegram_id}",
        },
        
        "metadata": {
            "user_id": user_id,
            "telegram_chat_id": chat_id,
        },
        "tags": ["telegram", "todo-bot"],
    }

    existing = await graph.aget_state(config)
    has_existing = bool(existing and existing.values)

    if has_existing:
        # Resume from checkpoint — only inject the new message
        # All other state (awaiting_conflict_confirmation, etc.) preserved
        input_state = {
            "messages": [HumanMessage(content=user_message)],
        }
    else:
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
            "search_results": "",
            "has_calendar_conflict": False,
            "calendar_conflicts": [],
            "calendar_check_done": False,
            "awaiting_conflict_confirmation": False,
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
