from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class TodoItem(TypedDict):
    id: str
    title: str
    description: Optional[str]
    due_date: Optional[str]
    priority: str   # low / medium / high
    status: str     # pending / in_progress / done
    tags: List[str]


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]
    user_id: str           # internal DB user id
    telegram_id: str       # telegram user id
    chat_id: str           # telegram chat id

    # Intent
    intent: str            # create / list / update / delete / remind / chat
    entities: dict         # extracted params from message

    # Working memory
    current_todos: List[TodoItem]
    last_action: str
    response_to_user: str

    confidence: float           # 0.0–1.0
    ambiguous_fields: List[str] # which fields are unclear

    search_results: str
