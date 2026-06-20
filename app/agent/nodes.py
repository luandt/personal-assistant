"""
Node functions for the LangGraph agent graph.
Each node receives AgentState and returns a partial state update.
"""
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Optional
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, merge_message_runs

from app.agent import state
from app.agent.prompt import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT, CHAT_RESPONSE_PROMPT
from app.agent.state import AgentState
from app.config import get_settings
from pydantic import BaseModel, Field, model_validator

from tavily import TavilyClient
from langgraph.store.base import BaseStore
from langchain_core.messages import ToolMessage
import dateparser, json
from zoneinfo import ZoneInfo
        
logger = logging.getLogger(__name__)
settings = get_settings()


class LLMConfigurationError(ValueError):
    """Raised when LLM provider/model configuration is invalid."""


def _build_chat_llm(provider: str, model_name: str):
    provider_name = (provider or "").strip().lower()
    kwargs = {"model": model_name, "temperature": 0, "max_tokens": 4096}

    try:
        if provider_name == "anthropic":
            from langchain_anthropic import ChatAnthropic

            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
            return ChatAnthropic(**kwargs)

        if provider_name == "openai":
            from langchain_openai import ChatOpenAI

            os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
            return ChatOpenAI(**kwargs)

        if provider_name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)
            return ChatGoogleGenerativeAI(**kwargs)

        if provider_name == "nvidia":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)
            if settings.nvidia_api_endpoint:
                os.environ.setdefault("NVIDIA_API_ENDPOINT", settings.nvidia_api_endpoint)
            return ChatNVIDIA(**kwargs)

        raise LLMConfigurationError(
            f"Unsupported llm_provider '{provider_name}'. Expected one of: nvidia, anthropic, openai, gemini"
        )
    except LLMConfigurationError:
        raise
    except ImportError as exc:
        raise LLMConfigurationError(
            f"Missing dependency for provider '{provider_name}'. Install required LangChain package."
        ) from exc
    except Exception as exc:
        raise LLMConfigurationError(
            f"Failed to initialize provider '{provider_name}' with model '{model_name}': {exc}"
        ) from exc

ALLOWED_INTENTS = {"create", "list", "update", "delete", "search", "chat", "web_search", "update_profile"}
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
PROFILE_NAMESPACE = "profile"
PROFILE_SIGNAL_PATTERNS = [
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi'm\b",
    r"\bim\b",
    r"\bi live in\b",
    r"\bi work as\b",
    r"\bi work in\b",
    r"\bi like\b",
    r"\bi love\b",
    r"\bi enjoy\b",
    r"\bi prefer\b",
    r"\bfavorite\b",
    r"\bmy hobby\b",
    r"\bmy hobbies\b",
    r"\bi'm into\b",
]


class Profile(BaseModel):
    """Structured profile memory for the user."""

    name: Optional[str] = Field(default=None, description="The user's name")
    location: Optional[str] = Field(default=None, description="The user's location")
    job: Optional[str] = Field(default=None, description="The user's job")
    connections: list[str] = Field(default_factory=list, description="Family, friends, or coworkers")
    interests: list[str] = Field(default_factory=list, description="Interests or hobbies")
    preferences: list[str] = Field(default_factory=list, description="User preferences")


def _has_profile_signals(message: str) -> bool:
    lowered = (message or "").lower()
    return any(re.search(pattern, lowered) for pattern in PROFILE_SIGNAL_PATTERNS)


def _format_profile_context(profile_items) -> str:
    if not profile_items:
        return ""

    lines = []
    for item in profile_items:
        value = item.value
        if isinstance(value, dict):
            parts = []
            for key in ("name", "location", "job", "connections", "interests", "preferences"):
                if value.get(key):
                    parts.append(f"{key}: {value[key]}")
            lines.append("; ".join(parts) if parts else json.dumps(value, ensure_ascii=False))
        else:
            lines.append(str(value))
    return "\n".join(lines)


def _first_clause(value: str) -> str:
    return re.split(r"[.?!]", value, maxsplit=1)[0].strip()


def _normalize_list_values(raw_value: str) -> list[str]:
    values = []
    for piece in re.split(r",|\band\b|&|/", raw_value, flags=re.IGNORECASE):
        cleaned = piece.strip().strip(".?!;:")
        cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


def _extract_profile_update(user_message: str) -> dict:
    message = (user_message or "").strip()
    if not message:
        return {}

    lowered = message.lower()
    update: dict[str, Any] = {}

    name_match = re.search(r"\bmy name is\s+(?P<value>.+)", lowered)
    if name_match:
        update["name"] = _first_clause(name_match.group("value")).title()

    location_match = re.search(r"\bi live in\s+(?P<value>.+)", lowered)
    if location_match:
        update["location"] = _first_clause(location_match.group("value")).title()

    job_match = re.search(r"\bi work as\s+(?P<value>.+)", lowered)
    if not job_match:
        job_match = re.search(r"\bi work in\s+(?P<value>.+)", lowered)
    if job_match:
        update["job"] = _first_clause(job_match.group("value")).strip()

    hobby_patterns = [
        r"\bmy hobby is\s+(?P<value>.+)",
        r"\bmy hobbies are\s+(?P<value>.+)",
        r"\bi like\s+(?P<value>.+)",
        r"\bi love\s+(?P<value>.+)",
        r"\bi enjoy\s+(?P<value>.+)",
        r"\bi(?:'|)m into\s+(?P<value>.+)",
    ]
    hobby_values: list[str] = []
    for pattern in hobby_patterns:
        match = re.search(pattern, lowered)
        if match:
            hobby_values.extend(_normalize_list_values(_first_clause(match.group("value"))))
    if hobby_values:
        update["interests"] = hobby_values

    preference_patterns = [
        r"\bi prefer\s+(?P<value>.+)",
        r"\bmy favorite (?P<label>.+?) is\s+(?P<value>.+)",
        r"\bmy favourite (?P<label>.+?) is\s+(?P<value>.+)",
    ]
    preference_values: list[str] = []
    for pattern in preference_patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        if "label" in match.groupdict():
            label = _first_clause(match.group("label")).strip()
            value = _first_clause(match.group("value")).strip()
            if label and value:
                preference_values.append(f"{label}: {value}")
        else:
            preference_values.extend(_normalize_list_values(_first_clause(match.group("value"))))
    if preference_values:
        update["preferences"] = preference_values

    return update


def _merge_profile(existing_profile: dict | None, profile_update: dict) -> dict:
    merged = dict(existing_profile or {})
    for key, value in profile_update.items():
        if value in (None, "", []):
            continue
        existing_value = merged.get(key)
        if isinstance(existing_value, list) and isinstance(value, list):
            merged[key] = existing_value + [item for item in value if item not in existing_value]
        else:
            merged[key] = value
    return merged


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


async def load_profile_context(store: BaseStore, user_id: str) -> str:
    """Load profile memory for prompt conditioning."""
    if not store or not user_id:
        return ""

    try:
        profile_item = await store.aget((PROFILE_NAMESPACE, user_id), "user_profile")
        logger.info(f"Loaded profile memory for user_id={user_id}: {profile_item}")
    except Exception as exc:
        logger.error(f"Profile memory load failed: {exc}")
        return ""

    return _format_profile_context([profile_item] if profile_item else [])
 
 
def make_nodes(tools: list, llm_model: str = None, llm_provider: str = None):
    """
    Returns node functions bound to the given tools and LLM.
    """
    model_name = llm_model or settings.llm_model
    provider_name = llm_provider or settings.llm_provider
    llm = _build_chat_llm(provider_name, model_name).bind_tools(tools)
    async def classify_intent(state: AgentState, store: BaseStore) -> dict:
        """
        Explicitly classify the user's intent and extract entities.
        Returns confidence score and detects ambiguity early.
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        awaiting = state.get("awaiting_conflict_confirmation", False)
        normalized_message = user_message.strip().lower()
        keep_confirmation_mode = awaiting and normalized_message in CONFIRM_REPLIES

        user_id = state.get("user_id", "")
        profile_context = await load_profile_context(store, user_id, )

        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_message=user_message, profile_context=profile_context or "none")
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
        user_id = state.get("user_id", "")

        profile_context = await load_profile_context(store, user_id)
        prompt = CHAT_RESPONSE_PROMPT
        if profile_context:
            prompt += f"\n\nUser profile memory:\n{profile_context}"
        prompt += f"\n\nUser: {user_message}\nPA:"
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

        # try:
        #     await write_memory(state, store, response.content)
        # except Exception as e:
        #     logger.warning(f"Memory write failed (non-blocking): {e}")

        return {
            "messages": [AIMessage(content=response.content)],
            "last_action": "chat_response_generated",
            "should_search_web": response.should_search_web,
            "entities": {"query": response.query} if response.should_search_web else {},
            "awaiting_conflict_confirmation": False,
        }

    async def update_profile_node(state: AgentState, store: BaseStore) -> dict:
        """Update profile memory from self-introductions and preference statements."""
        user_message = state["messages"][-1].content if state.get("messages") else ""
        user_id = state.get("user_id", "")

        if not user_message or not user_id:
            return {"last_action": "profile_memory_skipped"}

        if not _has_profile_signals(user_message):
            return {"last_action": "profile_memory_skipped"}

        profile_update = _extract_profile_update(user_message)
        if not profile_update:
            return {"last_action": "profile_memory_skipped"}

        namespace = (PROFILE_NAMESPACE, user_id)

        try:
            existing_item = await store.aget(namespace, "user_profile")
            existing_profile = existing_item.value if existing_item else None
        except Exception as exc:
            logger.error(f"Profile memory load failed: {exc}")
            existing_profile = None

        merged_profile = _merge_profile(existing_profile, profile_update)

        try:
            await store.aput(namespace, "user_profile", merged_profile)
        except Exception as exc:
            logger.warning(f"Profile memory update failed: {exc}")
            return {"last_action": "profile_memory_failed"}

        logger.info(f"Profile memory updated for user_id={user_id}: {list(profile_update.keys())}")
        return {"last_action": "profile_memory_updated"}
    
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
    return classify_intent, ask_clarification, todo_llm, execute_tools, web_search_node, chat_response_node, update_profile_node, check_calendar_node, ask_calendar_confirmation
 
 
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

    if intent == "update_profile":
        logger.info("Profile update intent detected")
        return "update_profile_node"
    
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