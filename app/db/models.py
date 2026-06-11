from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    String, Text, DateTime, Enum, ForeignKey, Boolean, JSON
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum
import uuid


class Base(DeclarativeBase):
    pass


class Priority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TodoStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_")
            alias_map = {
                "completed": "done",
                "complete": "done",
                "finished": "done",
                "in_progress": "in_progress",
                "in progress": "in_progress",
                "started": "in_progress",
                "doing": "in_progress",
                "pending": "pending",
                "todo": "pending",
                "open": "pending",
            }
            normalized = alias_map.get(normalized, normalized)
            if normalized in cls._value2member_map_:
                return cls(normalized)
        return super()._missing_(value)


def gen_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    telegram_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    preferences: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    todos: Mapped[List["Todo"]] = relationship("Todo", back_populates="user", cascade="all, delete-orphan")


class Todo(Base):
    __tablename__ = "todos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    priority: Mapped[Priority] = mapped_column(Enum(Priority), default=Priority.medium)
    status: Mapped[TodoStatus] = mapped_column(Enum(TodoStatus), default=TodoStatus.pending)
    tags: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    reminder_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now() )
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now() , onupdate=datetime.now() )

    user: Mapped["User"] = relationship("User", back_populates="todos")
