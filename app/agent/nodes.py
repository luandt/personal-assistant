"""
Node functions for the LangGraph agent graph.
Each node receives AgentState and returns a partial state update.
"""
from typing import Any
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a helpful personal assistant with access to a todo management system.

You can help users:
- Create todos, tasks, and reminders
- List and search their todos
- Update todo status, priority, and details
- Delete todos
- Set reminders with natural language dates

When using tools, always include the user_id from the conversation state.

Keep responses concise and friendly. Use the tools available to you and respond naturally.
After using tools, summarize what was done in a brief, human-friendly way.

Current user_id: {user_id}
"""

MAX_HISTORY_MESSAGES = 10


def make_nodes(tools: list, llm_model: str = None):
    """
    Returns node functions bound to the given tools and LLM.
    """
    model_name = llm_model or settings.llm_model
    llm = ChatNVIDIA(
        model=model_name,
        temperature=0,
    ).bind_tools(tools)

    async def call_llm(state: AgentState) -> dict:
        """Main LLM reasoning node — decides whether to use tools or respond directly."""
        user_id = state.get("user_id", "")
        system = SystemMessage(content=SYSTEM_PROMPT.format(user_id=user_id))
        messages = [system] + list(state["messages"])[-MAX_HISTORY_MESSAGES:]

        logger.info(f"Calling LLM with {len(messages)} messages (last {MAX_HISTORY_MESSAGES}), user_id={user_id}")
        response = await llm.ainvoke(messages)
        logger.info(f"LLM response type: {type(response).__name__}, has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"LLM tool calls: {[tc.get('name') for tc in response.tool_calls]}")
        else:
            logger.info(f"LLM response content: {response.content if hasattr(response, 'content') else response}")
        return {"messages": [response], "last_action": "llm_called"}

    async def execute_tools(state: AgentState) -> dict:
        """Execute any tool calls requested by the LLM."""
        from langchain_core.messages import ToolMessage
        last_message = state["messages"][-1]
        tool_results = []

        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning(f"execute_tools called but no tool_calls found. Message type: {type(last_message).__name__}")
            return {"messages": [], "last_action": "tools_executed"}

        logger.info(f"Executing {len(last_message.tool_calls)} tool calls")
        tool_map = {t.name: t for t in tools}
        logger.info(f"Available tools: {list(tool_map.keys())}")

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

            # Inject user_id if the tool needs it
            if "user_id" in tool_map.get(tool_name, {}).args if hasattr(tool_map.get(tool_name, {}), 'args') else {}:
                tool_args.setdefault("user_id", state.get("user_id", ""))

            # Ensure user_id is present for all todo tools
            if tool_name in {"create_todo", "list_todos", "update_todo", "delete_todo", "search_todos", "set_reminder"}:
                tool_args["user_id"] = state.get("user_id", "")

            try:
                result = await tool_map[tool_name].ainvoke(tool_args)
                logger.info(f"Tool '{tool_name}' result: {result}")
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {str(e)}", exc_info=True)
                result = f"❌ Tool error: {str(e)}"

            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_results, "last_action": "tools_executed"}

    return call_llm, execute_tools


def should_continue(state: AgentState) -> str:
    """Routing function: if last message has tool calls → execute them, else end."""
    messages = state["messages"]
    last_message = messages[-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    route = "execute_tools" if has_tool_calls else "end"
    logger.info(f"should_continue routing: last_message type={type(last_message).__name__}, has_tool_calls={has_tool_calls}, route={route}")
    return route
