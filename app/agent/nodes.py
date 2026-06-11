"""
Node functions for the LangGraph agent graph.
Each node receives AgentState and returns a partial state update.
"""
import json
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
 
CRITICAL EXECUTION RULE:
When you see "[Detected intent: ...]" in the system message, that means the user's intent 
and entities have already been classified and extracted. The confidence level tells you how sure we are.
 
If confidence >= 70%:
  → Use the provided entities and EXECUTE THE TOOL IMMEDIATELY
  → Do NOT ask "would you like me to...?" or "confirm the details?"
  → Example: If [Detected intent: create] with title="Meeting", due_date="Thursday",
             just call create_todo(title="Meeting", due_date="Thursday")
 
If confidence < 70%:
  → Ask clarifying questions about missing critical information
  
After executing tools, summarize what you did in a brief, friendly response.
 
When using tools, always include the user_id from the conversation state.
 
Current user_id: {user_id}
"""
 
INTENT_CLASSIFICATION_PROMPT = """Analyze this user message and extract the intent.
 
Return ONLY a valid JSON object, no other text:
{{
  "intent": "create|list|update|delete|search|chat",
  "confidence": 0.0-1.0,
  "entities": {{}},
  "ambiguous_fields": [],
  "clarification_question": null
}}
 
CRITICAL: For the "entities" field, extract these parameters if present in the user message:
 
FOR CREATE intent:
  - title: the task/meeting name (REQUIRED)
  - description: additional details (optional)
  - due_date_str: natural language date (e.g., "Thursday or Saturday this week", "tomorrow 3pm", "next Monday", "this Friday" means Friday this week) (optional)
  - priority: "low", "medium", or "high" (optional, default "medium")
  - tags: comma-separated tags (optional)
 
FOR LIST intent:
  - period: "today", "tomorrow", "week", or "all" (optional)
  - status: "pending", "in_progress", or "done" (optional)
  - priority: "low", "medium", or "high" (optional)
  - tags: comma-separated (optional)
 
FOR UPDATE/DELETE intents:
  - todo_id: the id of the task (or title fragment if available)
  - For UPDATE: also status, priority, due_date_str, tags as new values
 
FOR SEARCH intent:
  - query: the search keyword or phrase

FOR CHAT intent:
    - no specific entities, just general conversation
 
IMPORTANT RULES:
- confidence: 1.0 = 100% certain about what to do, 0.0 = completely unclear
- due_date_str: ALWAYS extract as natural language string, NEVER try to parse it
  Examples: "tomorrow 3pm", "Thursday this week", "next Monday", "in 2 days"
  The system will parse it using dateparser
- title: extract the exact task/meeting name from the message
- ambiguous_fields: list of fields you're uncertain about (e.g., ["todo_id", "due_date_str"])
- Return valid JSON only — no markdown, no code fences, no preamble
 
User message: {user_message}"""
 
 
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
        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(user_message=user_message)
        messages = [SystemMessage(content=prompt)]
        
        logger.info(f"Classifying intent for: {repr(user_message[:80])}")
        
        try:
            response = await llm.ainvoke(messages)
            text = extract_text(response.content)
            
            # Parse JSON response
            intent_data = json.loads(text)
            
            logger.info(
                f"Intent classified: {intent_data['intent']}, "
                f"confidence: {intent_data['confidence']:.2f}"
            )
            
            return {
                "intent": intent_data["intent"],
                "confidence": intent_data["confidence"],
                "entities": intent_data.get("entities", {}),
                "ambiguous_fields": intent_data.get("ambiguous_fields", []),
                "clarification_needed": intent_data.get("clarification_question") is not None,
                "clarification_question": intent_data.get("clarification_question"),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback: assume chat intent with low confidence
            return {
                "intent": "chat",
                "confidence": 0.3,
                "entities": {},
                "ambiguous_fields": ["all"],
                "clarification_needed": True,
                "clarification_question": "I'm not sure what you meant. Can you rephrase?",
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
        }
 
    async def todo_llm(state: AgentState) -> dict:
        """
        Main LLM reasoning node — uses tools to fulfill the request.
        Only called when intent is clear (confidence >= threshold).
        """
        user_id = state.get("user_id", "")
        system = SystemMessage(content=SYSTEM_PROMPT.format(user_id=user_id))
        messages = [system] + list(state["messages"])
 
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
 
    async def execute_tools(state: AgentState) -> dict:
        """Execute any tool calls requested by the LLM."""
        from langchain_core.messages import ToolMessage
        
        last_message = state["messages"][-1]
        tool_results = []
 
        tool_map = {t.name: t for t in tools}
 
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
 
            # Inject user_id if the tool needs it
            if tool_name in {"create_todo", "list_todos", "update_todo", "delete_todo", "search_todos", "set_reminder"}:
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
 
    return classify_intent, ask_clarification, todo_llm, execute_tools
 
 
def should_continue(state: AgentState) -> str:
    """
    Route after classify_intent:
    - If very low confidence (< 0.65) → ask_clarification
    - Otherwise → todo_llm (with classification info injected)
    
    This lets Claude handle minor ambiguities naturally while
    only asking for explicit clarification when truly uncertain.
    """
    confidence = state.get("confidence", 0.0)
    
    if confidence < 0.65:
        logger.info(f"Very low confidence ({confidence:.2f}) - asking for clarification")
        return "ask_clarification"
    
    logger.info(f"Adequate confidence ({confidence:.2f}) - proceeding to LLM")
    return "todo_llm"
 
 
def should_continue_llm(state: AgentState) -> str:
    """Route after todo_llm: if tool_calls → execute, else → end."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "end"
 