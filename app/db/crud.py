from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, delete
from app.db.models import User, Todo, Priority, TodoStatus


# ── Users ──────────────────────────────────────────────────────────────────

async def get_or_create_user(db: AsyncSession, telegram_id: str, username: str = None, first_name: str = None) -> User:
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(telegram_id=telegram_id, username=username, first_name=first_name)
        db.add(user)
        await db.flush()
    return user


async def get_user_by_telegram_id(db: AsyncSession, telegram_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    return result.scalar_one_or_none()


# ── Todos ──────────────────────────────────────────────────────────────────

async def create_todo(
    db: AsyncSession,
    user_id: str,
    title: str,
    description: str = None,
    due_date: datetime = None,
    priority: str = "medium",
    tags: list = None,
) -> Todo:
    todo = Todo(
        user_id=user_id,
        title=title,
        description=description,
        due_date=due_date,
        priority=Priority(priority),
        tags=tags or [],
    )
    db.add(todo)
    await db.flush()
    await db.refresh(todo)
    return todo


async def list_todos(
    db: AsyncSession,
    user_id: str,
    status: str = None,
    priority: str = None,
    tags: list = None,
    due_before: datetime = None,
    due_after: datetime = None,
    limit: int = 20,
) -> List[Todo]:
    conditions = [Todo.user_id == user_id]

    if status:
        conditions.append(Todo.status == TodoStatus(status))
    if priority:
        conditions.append(Todo.priority == Priority(priority))
    if due_before:
        conditions.append(Todo.due_date <= due_before)
    if due_after:
        conditions.append(Todo.due_date >= due_after)

    query = select(Todo).where(and_(*conditions)).order_by(Todo.due_date.asc().nulls_last(), Todo.created_at.desc()).limit(limit)
    result = await db.execute(query)
    todos = result.scalars().all()

    # Filter by tags if needed
    if tags:
        todos = [t for t in todos if any(tag in (t.tags or []) for tag in tags)]

    return todos


async def get_todo(db: AsyncSession, todo_id: str, user_id: str) -> Optional[Todo]:
    result = await db.execute(
        select(Todo).where(and_(Todo.id == todo_id, Todo.user_id == user_id))
    )
    return result.scalar_one_or_none()


async def update_todo(
    db: AsyncSession,
    todo_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Todo]:
    todo = await get_todo(db, todo_id, user_id)
    if todo is None:
        return None

    allowed_fields = {"title", "description", "due_date", "priority", "status", "tags", "reminder_sent"}
    for key, value in kwargs.items():
        if key in allowed_fields and value is not None:
            if key == "priority":
                value = Priority(value)
            elif key == "status":
                value = TodoStatus(value)
            setattr(todo, key, value)

    todo.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(todo)
    return todo


async def delete_todo(db: AsyncSession, todo_id: str, user_id: str) -> bool:
    todo = await get_todo(db, todo_id, user_id)
    if todo is None:
        return False
    await db.delete(todo)
    await db.flush()
    return True


async def delete_todos_by_tag(db: AsyncSession, user_id: str, tag: str) -> int:
    result = await db.execute(select(Todo).where(Todo.user_id == user_id))
    todos = result.scalars().all()
    count = 0
    for todo in todos:
        if tag in (todo.tags or []):
            await db.delete(todo)
            count += 1
    await db.flush()
    return count


async def search_todos(db: AsyncSession, user_id: str, query: str) -> List[Todo]:
    q = f"%{query.lower()}%"
    result = await db.execute(
        select(Todo).where(
            and_(
                Todo.user_id == user_id,
                or_(
                    Todo.title.ilike(q),
                    Todo.description.ilike(q),
                )
            )
        ).limit(10)
    )
    return result.scalars().all()


async def get_due_reminders(db: AsyncSession, before: datetime) -> List[Todo]:
    """Get todos with due_date <= before that haven't had reminders sent."""
    result = await db.execute(
        select(Todo).where(
            and_(
                Todo.due_date <= before,
                Todo.reminder_sent == False,
                Todo.status != TodoStatus.done,
            )
        )
    )
    return result.scalars().all()
