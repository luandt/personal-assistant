# Personal Assistant — Telegram Bot

A Telegram-based personal assistant with intelligent Todo management, powered by LangGraph + Claude.

---

## Stack

| Layer | Technology |
|---|---|
| Messaging | python-telegram-bot |
| API Server | FastAPI |
| Agent Framework | LangGraph |
| LLM | Anthropic / OpenAI / Gemini / NVIDIA (configurable) |
| Persistence | PostgreSQL + SQLAlchemy |
| Caching / Queue | Redis |
| Scheduler | APScheduler |
| Deployment | Docker Compose |

---

## Quick Start

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env with your keys
```

Required values in `.env`:
- `TELEGRAM_BOT_TOKEN` — from [@BotFather](https://t.me/BotFather)
- `TELEGRAM_WEBHOOK_URL` — your public HTTPS URL (e.g. from Railway/Fly.io)
- `LLM_PROVIDER` — one of: `anthropic`, `openai`, `gemini`, `nvidia`
- `LLM_MODEL` — provider-specific model name (example: `claude-sonnet-4-20250514`)
- Provider API key for the selected provider:
  - `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)
  - `OPENAI_API_KEY` — from [platform.openai.com](https://platform.openai.com)
  - `GEMINI_API_KEY` — from [aistudio.google.com](https://aistudio.google.com)
  - `NVIDIA_API_KEY` — from [build.nvidia.com](https://build.nvidia.com)

### 2. Run with Docker Compose

```bash
docker compose up --build
```

This starts:
- **FastAPI app** on port 8000
- **PostgreSQL** on port 5432
- **Redis** on port 6379

### 3. Expose to internet (for Telegram webhooks)

For local dev, use [ngrok](https://ngrok.com):

```bash
ngrok http 8000
# Copy the https URL into TELEGRAM_WEBHOOK_URL in .env
```

### 4. Talk to your bot!

Open Telegram and message your bot. Examples:

```
Add buy groceries tomorrow evening
What do I have this week?
Remind me to call mom tomorrow 3pm
Mark gym as done
Delete everything tagged #work
Search for dentist
```

---

## Project Structure

```
personal-assistant/
├── app/
│   ├── main.py                  # FastAPI entrypoint & lifespan
│   ├── config.py                # Settings (pydantic-settings)
│   ├── telegram/
│   │   ├── webhook.py           # Receive & route Telegram messages
│   │   └── sender.py            # Send responses back
│   ├── agent/
│   │   ├── graph.py             # LangGraph graph definition
│   │   ├── nodes.py             # LLM call + tool executor nodes
│   │   ├── tools.py             # Todo CRUD tools
│   │   └── state.py             # AgentState schema
│   ├── db/
│   │   ├── models.py            # SQLAlchemy models (User, Todo)
│   │   ├── session.py           # Engine & session factory
│   │   └── crud.py              # DB operations
│   └── scheduler/
│       └── reminders.py         # APScheduler reminder jobs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Agent Flow

```
User message → FastAPI webhook
  → get_or_create_user
  → LangGraph agent (thread per user, checkpointed in Postgres)
      → Configured LLM provider reasons + decides tools
      → Executes todo tools (create / list / update / delete / search / remind)
      → Loops until done
      → Returns natural language response
  → Send reply via Telegram
```

---

## Available Tools

| Tool | What it does |
|---|---|
| `create_todo` | Create a new todo with optional due date, priority, tags |
| `list_todos` | List todos filtered by status, priority, tags, or period |
| `update_todo` | Update title, status, priority, due date, or tags |
| `delete_todo` | Delete by ID or bulk-delete by tag |
| `search_todos` | Full-text search across title and description |
| `set_reminder` | Set/update the reminder time for a todo |

---

## Roadmap

- **MVP (Week 1–2)** ✅ Webhook + CRUD tools + LangGraph skeleton + reminders
- **Week 3** — Long-term memory (user preferences), fuzzy name matching
- **Week 4** — Priority/tag management, snooze reminders, polish
- **Future** — Calendar sync, habit tracking, voice notes
