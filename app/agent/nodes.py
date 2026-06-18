"""
Node functions for the LangGraph agent graph.
Each node receives AgentState and returns a partial state update.
"""
import json
from typing import Any
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.agent import state
from app.agent.prompt import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT, CHAT_RESPONSE_PROMPT
from app.agent.state import AgentState
from app.config import get_settings

from tavily import TavilyClient
from langgraph.store.base import BaseStore
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field, model_validator
import dateparser, json
from datetime import timedelta
from zoneinfo import ZoneInfo
        
logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_INTENTS = {"create", "list", "update", "delete", "search", "chat", "web_search"}
CONFIRM_POSITIVE = {"yes", "y", "yeah", "yep", "ok", "sure", "confirm"}
CONFIRM_NEGATIVE = {"no", "n", "nope", "cancel", "stop"}
CONFIRM_REPLIES = CONFIRM_POSITIVE | CONFIRM_NEGATIVE
TODO_TOOL_NAMES = {
    "create_todo",
    "list_todos",
    "update_todo",
    "delete_todo",
    "search_todos",
    "set_reminder",
}
CALENDAR_READ_ONLY_TOOL_NAMES = {
    "list-calendars",
    "list-events",
    "search-events",
    "get-event",
    "list-colors",
    "get-freebusy",
    "get-current-time",
}
CALENDAR_TIMEZONE = "Asia/Ho_Chi_Minh"
CALENDAR_TZ = ZoneInfo(CALENDAR_TIMEZONE)


def _fallback_intent_response() -> dict:
    return {
        "intent": "chat",
        "confidence": 0.3,
        "entities": {},
        "ambiguous_fields": ["all"],
        "clarification_needed": True,
        "clarification_question": "I'm not sure what you meant. Can you rephrase?",
    }


def _validate_intent_payload(intent_data: dict[str, Any]) -> dict:
    intent = intent_data.get("intent")
    confidence = intent_data.get("confidence")

    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise ValueError(f"Invalid intent: {intent}")

    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Invalid confidence type: {type(confidence)}")
    confidence = max(0.0, min(1.0, float(confidence)))

    entities = intent_data.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}

    # Enforce minimum required field for create intent.
    if intent == "create" and not entities.get("title"):
        raise ValueError("Missing required entity for create intent: title")

    ambiguous_fields = intent_data.get("ambiguous_fields") or []
    if not isinstance(ambiguous_fields, list):
        ambiguous_fields = ["all"]

    clarification_question = intent_data.get("clarification_question")

    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "ambiguous_fields": ambiguous_fields,
        "clarification_needed": clarification_question is not None,
        "clarification_question": clarification_question,
    }

class ChatResponse(BaseModel):
    should_search_web: bool = False
    content: str = Field(default="", description="The assistant's reply to the user")
    query: str = Field(default="", description="The search query for web search, if applicable")

    @model_validator(mode="after")
    def force_search_if_query_extracted(self) -> "ChatResponse":
        """If the LLM extracted a search query, force should_search_web to True."""
        if self.query.strip():
            self.should_search_web = True
            self.content = ""
        return self

def extract_text(content) -> str:
    """
    Safely extract plain text from an AIMessage content field.
    Claude returns either:
      - a plain string:  "Here are your todos..."
      - a list of blocks: [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content).strip()
 
 
def make_nodes(tools: list, llm_model: str = None):
    """
    Returns node functions bound to the given tools and LLM.
    """
    model_name = llm_model or settings.llm_model
    llm = ChatNVIDIA(
        model=model_name,
        temperature=0,
        max_tokens=4096,
    ).bind_tools(tools)
 
    async def classify_intent(state: AgentState) -> dict:
        """
        Explicitly classify the user's intent and extract entities.
        Returns confidence score and detects ambiguity early.
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        awaiting = state.get("awaiting_conflict_confirmation", False)
        normalized_message = user_message.strip().lower()
        keep_confirmation_mode = awaiting and normalized_message in CONFIRM_REPLIES
        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_message=user_message)
        messages = [SystemMessage(content=prompt)]
        
        logger.info(f"Classifying intent for: {repr(user_message[:80])}")
        
        try:
            response = await llm.ainvoke(messages)
            text = extract_text(response.content)
            
            # Parse JSON response
            intent_data = json.loads(text)
            validated = _validate_intent_payload(intent_data)
            
            logger.info(
                f"Intent classified: {validated['intent']}, "
                f"confidence: {validated['confidence']:.2f}"
            )

            return {
                **validated,
                "awaiting_conflict_confirmation": keep_confirmation_mode,
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback: assume chat intent with low confidence
            return {
                **_fallback_intent_response(),
                "awaiting_conflict_confirmation": keep_confirmation_mode,
            }
 
    async def ask_clarification(state: AgentState) -> dict:
        """
        When intent is ambiguous, generate clarification options
        and ask the user to choose.
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        intent = state.get("intent", "")
        question = state.get("clarification_question", "Can you clarify what you meant?")
        
        logger.info(f"Asking for clarification (intent={intent})")
        
        # Build a friendly clarification message
        response_text = f"I'm not sure what you meant by '{user_message}'.\n\n{question}\n\nPlease respond with your choice or rephrase."
        
        return {
            "messages": [AIMessage(content=response_text)],
            "last_action": "clarification_asked",
            "awaiting_conflict_confirmation": False,

        }
 
    async def todo_llm(state: AgentState) -> dict:
        """
        Main LLM reasoning node — uses tools to fulfill the request.
        Only called when intent is clear (confidence >= threshold).
        """
        user_id = state.get("user_id", "")
        system = SystemMessage(content=SYSTEM_PROMPT.format(user_id=user_id))
        messages = [system] + list(state["messages"])[-10:]
 
        logger.info(
            f"Calling LLM with {len(messages)} messages, "
            f"intent={state.get('intent')}, "
            f"user_id={user_id}"
        )
        response = await llm.ainvoke(messages)
 
        tool_calls = getattr(response, "tool_calls", [])
        text = extract_text(response.content)
        logger.info(
            f"LLM response: intent={state.get('intent')}, "
            f"has_tool_calls={bool(tool_calls)}, "
            f"tools=[{','.join(t['name'] for t in tool_calls)}]"
        )
 
        return {
            "messages": [response],
            "last_action": "llm_called",
        }
    
    async def web_search_node(state: AgentState) -> dict:
        """
        Perform a web search using Tavily API for non-todo queries.
        Returns formatted search results.
        """
        
        user_message = state["messages"][-1].content
        query = state.get("entities", {}).get("query", user_message)
        
        logger.info(f"Web search query: {repr(query[:80])}")
        
        try:
            client = TavilyClient(api_key=settings.tavily_api_key)
            
            # Perform search
            results = client.search(
                query=query,
                include_domains=[],
                max_results=5,  # Get top 5 results
                include_answer=True  # Include AI-generated answer
            )
            
            response_text = format_tavily_results(query, results)
            
            logger.info(f"Web search completed: {len(results.get('results', []))} results")
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            response_text = f"Sorry, I couldn't search the web right now. Error: {str(e)}"
        
        return {
            "messages": [AIMessage(content=response_text)],
            "last_action": "web_search_completed",
            "search_results": response_text,
        }
 
    async def chat_response_node(state: AgentState, store: BaseStore) -> dict:
        """
        Generate a response for general chat messages (non-todo).
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        prompt = CHAT_RESPONSE_PROMPT + f"\n\nUser: {user_message}\nPA:"
        messages = [SystemMessage(content=prompt)]
        
        logger.info(f"Generating chat response for: {repr(user_message[:80])}")

        structured_llm = llm.with_structured_output(ChatResponse)
        try:
            response = await structured_llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"Structured chat response failed: {e}")
            fallback_content = "Sorry, I had trouble generating a response. Could you rephrase that?"
            return {
                "messages": [AIMessage(content=fallback_content)],
                "last_action": "chat_response_fallback",
                "should_search_web": False,
                "entities": {},
            }

        try:
            await write_memory(state, store, response.content)
        except Exception as e:
            logger.warning(f"Memory write failed (non-blocking): {e}")

        return {
            "messages": [AIMessage(content=response.content)],
            "last_action": "chat_response_generated",
            "should_search_web": response.should_search_web,
            "entities": {"query": response.query} if response.should_search_web else {},
            "awaiting_conflict_confirmation": False,
        }
    
    async def execute_tools(state: AgentState) -> dict:
        """Execute any tool calls requested by the LLM."""
        
        last_message = state["messages"][-1]
        tool_results = []
 
        tool_map = {t.name: t for t in tools}
 
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name not in tool_map:
                logger.warning(f"Tool {tool_name} not found in available tool map")
                result = "❌ Tool not available in this environment."
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            is_calendar_read_tool = tool_name in CALENDAR_READ_ONLY_TOOL_NAMES
            is_todo_tool = tool_name in TODO_TOOL_NAMES
            # Any MCP calendar tool outside the read-only list is treated as write/restricted.
            is_calendar_write_tool = ("-" in tool_name) and not is_calendar_read_tool

            if is_calendar_write_tool:
                logger.warning(
                    f"Blocked restricted calendar tool call: {tool_name}"
                )
                result = "❌ Calendar write tools are disabled. I can only create todos in Postgres."
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if state.get("intent") == "create" and not (is_todo_tool or is_calendar_read_tool):
                logger.warning(
                    f"Blocked non-todo tool for create intent: {tool_name}"
                )
                result = "❌ For create requests, only todo creation and calendar conflict checks are allowed."
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
 
            # Inject user_id if the tool needs it
            if tool_name in TODO_TOOL_NAMES:
                tool_args["user_id"] = state.get("user_id", "")
 
            try:
                result = await tool_map[tool_name].ainvoke(tool_args)
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                result = f"❌ Tool error: {str(e)}"
 
            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )
 
        return {
            "messages": tool_results,
            "last_action": "tools_executed",
        }
 
    async def check_calendar_node(state: AgentState) -> dict:
        intent = state.get("intent", "")
        entities = state.get("entities", {})
        due_date_str = entities.get("due_date_str", "")

        if intent != "create" or not due_date_str:
            return {
                "has_calendar_conflict": False,
                "calendar_conflicts": [],
                "calendar_check_done": True,
            }

        parsed_dt = dateparser.parse(
            due_date_str,
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": CALENDAR_TIMEZONE,
                "RETURN_AS_TIMEZONE_AWARE": True,
            }
        )
        if not parsed_dt:
            logger.warning(f"Could not parse date: {due_date_str}")
            return {
                "has_calendar_conflict": False,
                "calendar_conflicts": [],
                "calendar_check_done": True,
            }

        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=CALENDAR_TZ)
        else:
            parsed_dt = parsed_dt.astimezone(CALENDAR_TZ)

        start_iso = parsed_dt.isoformat()
        end_iso = (parsed_dt + timedelta(hours=1)).isoformat()

        logger.info(f"Checking calendar: {start_iso} → {end_iso}")

        freebusy_tool = next((t for t in tools if t.name == "get-freebusy"), None)
        list_events_tool = next((t for t in tools if t.name == "list-events"), None)

        has_conflict = False
        conflict_events = []

        # ── Step 1: get-freebusy (catches timed events) ──────────────
        if freebusy_tool:
            try:
                result = await freebusy_tool.ainvoke({
                    "calendars": [{"id": "primary"}],
                    "timeMin": start_iso,
                    "timeMax": end_iso,
                    "timeZone": CALENDAR_TIMEZONE,
                })

                text = next(
                    (b["text"] for b in result
                    if isinstance(b, dict) and b.get("type") == "text"),
                    ""
                ) if isinstance(result, list) else str(result)

                data = json.loads(text)
                for cal_id, cal_data in data.get("calendars", {}).items():
                    busy = cal_data.get("busy", [])
                    if busy:
                        has_conflict = True
                        conflict_events = [
                            {"title": "Busy", "start": b["start"], "end": b["end"], "all_day": False}
                            for b in busy
                        ]
                        break

                logger.info(f"get-freebusy: conflict={has_conflict}")

            except Exception as e:
                logger.error(f"get-freebusy failed: {e}")

        # ── Step 2: list-events detail enrichment (adds titles/all-day info) ────
        if list_events_tool:
            try:
                result = await list_events_tool.ainvoke({
                    "calendarId": "primary",
                    "timeMin": start_iso,
                    "timeMax": end_iso,
                    "timeZone": CALENDAR_TIMEZONE,
                })

                text = next(
                    (b["text"] for b in result
                    if isinstance(b, dict) and b.get("type") == "text"),
                    ""
                ) if isinstance(result, list) else str(result)

                data = json.loads(text)
                events = data if isinstance(data, list) else data.get("events", [])

                if events:
                    has_conflict = True
                    conflict_events = [
                        {
                            "title": e.get("summary", "Busy"),
                            "start": e.get("start", {}).get("dateTime")
                                    or e.get("start", {}).get("date", ""),
                            "end": e.get("end", {}).get("dateTime")
                                or e.get("end", {}).get("date", ""),
                            "all_day": "date" in e.get("start", {}),
                        }
                        for e in events
                    ]
                    logger.info(
                        f"list-events: found {len(events)} event(s) — using detailed conflict events"
                    )
                else:
                    logger.info("list-events: no events found — keeping freebusy conflict details")

            except Exception as e:
                logger.error(f"list-events failed: {e}")

        logger.info(f"Final conflict={has_conflict}, events={conflict_events}")

        return {
            "has_calendar_conflict": has_conflict,
            "calendar_conflicts": conflict_events,
            "calendar_check_done": True,
            "awaiting_conflict_confirmation": False,

        }

    async def ask_calendar_confirmation(state: AgentState) -> dict:
        """
        Inform user about conflict and ask if they want to proceed.
        """
        from langchain_core.messages import AIMessage

        entities = state.get("entities", {})
        conflicts = state.get("calendar_conflicts", [])
        title = entities.get("title", "the task")
        due_date = entities.get("due_date_str", "the requested time")

        conflict_detail = ""
        if conflicts:
            lines = []
            for conflict in conflicts[:3]:
                conflict_title = conflict.get("title", "Busy")
                start = conflict.get("start", "")
                end = conflict.get("end", "")
                if conflict.get("all_day"):
                    lines.append(f"- {conflict_title} (all day: {start} to {end})")
                elif start and end:
                    lines.append(f"- {conflict_title} ({start} to {end})")
                else:
                    lines.append(f"- {conflict_title}")

            if lines:
                conflict_detail = "\n\nConflicting calendar event(s):\n" + "\n".join(lines)

        message = (
            f"⚠️ *Calendar Conflict Detected!*\n\n"
            f"You have an existing event that overlaps with:\n"
            f"📌 *{title}* at *{due_date}*"
            f"{conflict_detail}\n\n"
            f"Do you still want to create this task? "
            f"Reply *yes* to confirm or *no* to cancel."
        )

        return {
            "messages": [AIMessage(content=message)],
            "last_action": "calendar_confirmation_asked",
            "awaiting_conflict_confirmation": True,
        }
    return classify_intent, ask_clarification, todo_llm, execute_tools, web_search_node, chat_response_node, check_calendar_node, ask_calendar_confirmation
 
 
def messages_router(state: AgentState) -> str:
    """
    Route after classify_intent:
    - If web_search intent → web_search_node
    - If very low confidence → ask_clarification
    - Otherwise → todo_llm
    """
    intent = state.get("intent", "")
    confidence = state.get("confidence", 0.0)

    awaiting = state.get("awaiting_conflict_confirmation", False)
    logger.info(f"Routing after intent classification: intent={intent}, awaiting_confirmation={awaiting}")

    # User is confirming/cancelling a conflict warning
    if awaiting:
        last_msg = state["messages"][-1].content.strip().lower()
        if last_msg in CONFIRM_POSITIVE:
            logger.info("User confirmed calendar conflict — proceeding to create")
            return "todo_llm"

        if last_msg in CONFIRM_NEGATIVE:
            logger.info("User declined calendar conflict — switching to chat response")
            return "chat_response_node"

        logger.info("Stale/irrelevant confirmation state detected — routing by intent")
    
    if confidence < 0.65:
        logger.info(f"Low confidence ({confidence:.2f}) - asking for clarification")
        return "ask_clarification"
    
    if intent == "create":
        return "check_calendar"
    
    if intent == "chat":
        logger.info("Chat intent detected")
        return "chat_response_node"
    
    if intent == "web_search":
        logger.info("Web search intent detected")
        return "web_search"
    
    logger.info(f"Adequate confidence ({confidence:.2f}) - proceeding to LLM")
    return "todo_llm"
 
 
def should_execute_tools(state: AgentState) -> str:
    """Route after todo_llm: if tool_calls → execute, else → end."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "end"

def should_search_web_after_chat(state: AgentState) -> str:
    """After a chat response, check if we should do a web search."""
    if state.get("should_search_web"):
        logger.info(f"Routing to web search with query: {state.get('entities', {}).get('query', '')}")
        return "web_search"
    return "end" 

def should_continue_after_calendar(state: AgentState) -> str:
    """Route after calendar check: conflict → confirm, else → todo_llm."""
    if state.get("has_calendar_conflict", False):
        return "ask_calendar_confirmation"
    return "todo_llm"

def format_tavily_results(query: str, results: dict) -> str:
    """
    Format Tavily search results into readable text.
    
    Tavily returns:
    {
        "answer": "AI-generated answer",
        "results": [
            {
                "title": "Result title",
                "url": "https://...",
                "content": "Result content",
                "score": 0.95
            },
            ...
        ]
    }
    """
    lines = [f"🔍 Search results for: **{query}**\n"]
    
    # Include AI answer if available
    if results.get("answer"):
        lines.append(f"📝 Summary: {results['answer']}\n")
    
    # Include top results
    if results.get("results"):
        lines.append("📋 Sources:")
        for i, result in enumerate(results["results"][:3], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "No content")[:150]
            
            lines.append(f"\n{i}. **{title}**")
            if url:
                lines.append(f"   🔗 {url}")
            lines.append(f"   {content}...")
    else:
        lines.append("No results found.")
    
    return "\n".join(lines)

async def write_memory(state: AgentState, store: BaseStore, response_content: str):
    """ Store chat response"""
    user_id = state.get("user_id", "")
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = await store.aget(namespace, key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."
    
    await store.aput(namespace, key, {"memory": response_content})