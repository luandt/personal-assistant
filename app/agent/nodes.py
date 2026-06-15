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

from tavily import TavilyClient
from langgraph.store.base import BaseStore
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field, model_validator
        
logger = logging.getLogger(__name__)
settings = get_settings()

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

FOR WEB_SEARCH intent (web search):
  - query: the web search question as-is
  Example: "What's the weather in Ho Chi Minh City?" → 
    {{"intent": "web_search", "query": "weather Ho Chi Minh City", "confidence": 0.95}}
  Example: "Who won World Cup 2024?" → 
    {{"intent": "web_search", "query": "World Cup 2024 winner", "confidence": 0.92}}

DETECTION RULES FOR WEB_SEARCH:
    - Message must be start with: "Search web for", "Search for", "Look up", "Find out", "What's", "Who", "How do I", "What are the latest", "google", "bing", "tavily"
    - If not start with above keywords, it's possible a genenral chat message, even if it contains a question. In that case, classify as "chat" intent, not "web_search".
    - Questions about current events, news, weather, sports 
    
    scores
    - How-to questions about non-todo topics
    - Questions starting with: "What's", "Who", "How do I", "What are the latest"
    - Not recommend about todo topics (e.g., "Do you want to create a todo/reminder for that?") because it can be a general chat message

IMPORTANT RULES:
- confidence: 1.0 = 100% certain about what to do, 0.0 = completely unclear
- due_date_str: ALWAYS extract as natural language string, NEVER try to parse it
  Examples: "tomorrow 3pm", "Thursday this week", "next Monday", "in 2 days"
  The system will parse it using dateparser
- title: extract the exact task/meeting name from the message
- ambiguous_fields: list of fields you're uncertain about (e.g., ["todo_id", "due_date_str"])
- Return valid JSON only — no markdown, no code fences, no preamble
 
User message: {user_message}"""

CHAT_RESPONSE_PROMPT = """You are PA, a helpful chat assistant.

Important:
- Do NOT mention todo, reminders, task creation, or scheduling unless the user clearly asks for it.
- For normal chat, answer naturally.
- Answer in the language the user used.
- If you do not understand the user's message, ask them to rephrase.
- If the user expresses sadness, fear, death, or hopelessness, respond supportively and ask a gentle follow-up.

Set should_search_web to TRUE if the user asks about ANY of the following:
- Sports scores, match results, or game outcomes
- Recent news or current events
- Stock prices, crypto, or financial data
- Weather
- Anything that requires up-to-date or real-time information
- Anything you are uncertain or unsure about

IMPORTANT: If should_search_web is true, set content to an empty string "". Do NOT say things like 
"I can't find that" or "let me look that up" — just set should_search_web to true and let the search handle it.

You must always respond in the following JSON format:
{
  "content": "<your reply, or empty string if should_search_web is true>",
  "should_search_web": <true or false>,
  "query": "<optimized search query if should_search_web is true, otherwise empty string>"
}"""

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
    
    async def web_search_node(state: AgentState) -> dict:
        """
        Perform a web search using Tavily API for non-todo queries.
        Returns formatted search results.
        """
        
        user_message = state["messages"][-1].content
        query = state.get("entities", {}).get("query", user_message)
        
        logger.info(f"Web search query: {repr(query[:80])}")
        
        try:
            # Initialize Tavily client with API key
            client = TavilyClient(api_key=settings.tavily_api_key)
            
            # Perform search
            results = client.search(
                query=query,
                include_domains=[],
                max_results=5,  # Get top 5 results
                include_answer=True  # Include AI-generated answer
            )
            
            # Format results for user
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
        response = await structured_llm.ainvoke(messages)
        
        await write_memory(state, store, response.content)
        return {
            "messages": [AIMessage(content=response.content)],
            "last_action": "chat_response_generated",
            "should_search_web": response.should_search_web,
            "entities": {"query": response.query} if response.should_search_web else {},

        }
    
    async def execute_tools(state: AgentState) -> dict:
        """Execute any tool calls requested by the LLM."""
        
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
 
    return classify_intent, ask_clarification, todo_llm, execute_tools, web_search_node, chat_response_node
 
 
def should_continue(state: AgentState) -> str:
    """
    Route after classify_intent:
    - If web_search intent → web_search_node
    - If very low confidence → ask_clarification
    - Otherwise → todo_llm
    """
    intent = state.get("intent", "")
    confidence = state.get("confidence", 0.0)
    
    if intent == "chat":
        logger.info("Chat intent detected")
        return "chat_response_node"
    
    if intent == "web_search":
        logger.info("Web search intent detected")
        return "web_search"
    
    if confidence < 0.65:
        logger.info(f"Low confidence ({confidence:.2f}) - asking for clarification")
        return "ask_clarification"
    
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
    last_message = state["messages"][-1]
    if state.get("should_search_web"):
        print(f"Routing to web search with query: {state.get('entities', {}).get('query', '')}")
        return "web_search"
    return "end" 

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