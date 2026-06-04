"""
LangGraph tools — each tool receives the DB session via a closure created
at graph-build time (inject_db_tools), so the graph remains pure Python
with no FastAPI dependency injection needed.
"""
from datetime import datetime, timedelta
from typing import Optional, List
import dateparser
from langchain_core.tools import tool


def make_todo_tools(db_session_factory):
    """
    Returns a list of LangGraph-compatible tools bound to an async DB session factory.
    Call once at startup and pass tools into the graph builder.
    """

    @tool
    async def create_todo(
        title: str,
        user_id: str,
        description: str = "",
        due_date_str: str = "",
        priority: str = "medium",
        tags: str = "",
    ) -> str:
        """Create a new todo item. due_date_str accepts natural language like 'tomorrow 3pm'."""
        from app.db import crud
        due_date = None
        if due_date_str:
            due_date = dateparser.parse(due_date_str, settings={"PREFER_DATES_FROM": "future"})

        tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else []

        async with db_session_factory() as db:
            todo = await crud.create_todo(
                db,
                user_id=user_id,
                title=title,
                description=description or None,
                due_date=due_date,
                priority=priority,
                tags=tag_list,
            )
            await db.commit()
            due_str = todo.due_date.strftime("%Y-%m-%d %H:%M") if todo.due_date else "no due date"
            return f"✅ Created todo [{todo.id[:8]}]: '{todo.title}' | priority: {todo.priority.value} | due: {due_str} | tags: {todo.tags}"

    @tool
    async def list_todos(
        user_id: str,
        status: str = "",
        priority: str = "",
        tags: str = "",
        period: str = "",
    ) -> str:
        """List todos for a user. period must be one of: 'today', 'tomorrow', 'week', 'all'. Use 'tomorrow' when user asks about tomorrow's tasks. Use 'today' for today. Use 'week' for next 7 days. Use 'all' to list everything."""
        from app.db import crud
        due_before = None
        due_after = None
        now = datetime.utcnow()
 
        if period == "today":
            due_after = now.replace(hour=0, minute=0, second=0)
            due_before = now.replace(hour=23, minute=59, second=59)
        elif period == "tomorrow":
            tomorrow = now + timedelta(days=1)
            due_after = tomorrow.replace(hour=0, minute=0, second=0)
            due_before = tomorrow.replace(hour=23, minute=59, second=59)
        elif period == "week":
            due_after = now
            due_before = now + timedelta(days=7)

        tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else None

        async with db_session_factory() as db:
            todos = await crud.list_todos(
                db,
                user_id=user_id,
                status=status or None,
                priority=priority or None,
                tags=tag_list,
                due_before=due_before,
                due_after=due_after,
            )

        if not todos:
            return "📭 No todos found matching your criteria."

        lines = ["📋 Your todos:\n"]
        for t in todos:
            due_str = t.due_date.strftime("%b %d %H:%M") if t.due_date else "—"
            tag_str = " ".join(f"#{tag}" for tag in (t.tags or []))
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "done": "✅"}.get(t.status.value, "•")
            priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(t.priority.value, "•")
            lines.append(
                f"{status_emoji} [{t.id[:8]}] {t.title}\n"
                f"   status: {t.status.value} | priority: {t.priority.value} | 📅 {due_str} {tag_str}"
            )

        return "\n".join(lines)

    @tool
    async def update_todo(
        user_id: str,
        todo_id: str,
        title: str = "",
        status: str = "",
        priority: str = "",
        due_date_str: str = "",
        tags: str = "",
    ) -> str:
        """Update an existing todo. Provide todo_id (at least 8 chars) and the fields to change."""
        from app.db import crud
        due_date = None
        if due_date_str:
            due_date = dateparser.parse(due_date_str, settings={"PREFER_DATES_FROM": "future"})

        tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else None

        async with db_session_factory() as db:
            # Support partial ID match
            todos = await crud.list_todos(db, user_id=user_id, limit=100)
            matched = [t for t in todos if t.id.startswith(todo_id) or t.id == todo_id]
            if not matched:
                return f"❌ No todo found with id starting with '{todo_id}'"
            todo = matched[0]

            updated = await crud.update_todo(
                db,
                todo_id=todo.id,
                user_id=user_id,
                title=title or None,
                status=status or None,
                priority=priority or None,
                due_date=due_date,
                tags=tag_list,
            )
            await db.commit()
            if updated is None:
                return f"❌ Could not update todo '{todo_id}'"
            return f"✏️ Updated todo [{updated.id[:8]}]: '{updated.title}' → status: {updated.status.value}"

    @tool
    async def delete_todo(
        user_id: str,
        todo_id: str = "",
        tag: str = "",
    ) -> str:
        """Delete a todo by ID, or all todos with a specific tag."""
        from app.db import crud
        async with db_session_factory() as db:
            if tag:
                tag_clean = tag.lstrip("#")
                count = await crud.delete_todos_by_tag(db, user_id=user_id, tag=tag_clean)
                await db.commit()
                return f"🗑️ Deleted {count} todo(s) tagged #{tag_clean}"
            elif todo_id:
                todos = await crud.list_todos(db, user_id=user_id, limit=100)
                matched = [t for t in todos if t.id.startswith(todo_id) or t.id == todo_id]
                if not matched:
                    return f"❌ No todo found with id starting with '{todo_id}'"
                ok = await crud.delete_todo(db, todo_id=matched[0].id, user_id=user_id)
                await db.commit()
                return f"🗑️ Deleted todo '{matched[0].title}'" if ok else "❌ Delete failed"
            else:
                return "❌ Provide either todo_id or tag to delete."

    @tool
    async def search_todos(user_id: str, query: str) -> str:
        """Search todos by keyword in title or description."""
        from app.db import crud
        async with db_session_factory() as db:
            todos = await crud.search_todos(db, user_id=user_id, query=query)

        if not todos:
            return f"🔍 No todos found matching '{query}'"

        lines = [f"🔍 Found {len(todos)} todo(s) matching '{query}':\n"]
        for t in todos:
            due_str = t.due_date.strftime("%b %d %H:%M") if t.due_date else "—"
            lines.append(f"• [{t.id[:8]}] {t.title} | {t.status.value} | due: {due_str}")
        return "\n".join(lines)

    @tool
    async def set_reminder(
        user_id: str,
        todo_id: str,
        remind_at_str: str,
    ) -> str:
        """Set or update a reminder for a todo. remind_at_str accepts natural language."""
        from app.db import crud
        remind_at = dateparser.parse(remind_at_str, settings={"PREFER_DATES_FROM": "future"})
        if not remind_at:
            return f"❌ Could not parse reminder time: '{remind_at_str}'"

        async with db_session_factory() as db:
            todos = await crud.list_todos(db, user_id=user_id, limit=100)
            matched = [t for t in todos if t.id.startswith(todo_id) or t.id == todo_id]
            if not matched:
                return f"❌ No todo found with id starting with '{todo_id}'"
            todo = matched[0]

            updated = await crud.update_todo(
                db,
                todo_id=todo.id,
                user_id=user_id,
                due_date=remind_at,
                reminder_sent=False,
            )
            await db.commit()

        return f"⏰ Reminder set for '{updated.title}' at {remind_at.strftime('%b %d %H:%M')}"

    return [create_todo, list_todos, update_todo, delete_todo, search_todos, set_reminder]
