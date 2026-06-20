# Personal Assistant — Telegram Bot

A Telegram-based personal assistant with intelligent Todo management, profile-aware web search, and configurable LLM providers, powered by LangGraph.

---

## Stack

| Layer | Technology |
|---|---|
| Messaging | python-telegram-bot |
| API Server | FastAPI |
| Agent Framework | LangGraph |
| LLM | Anthropic / OpenAI / Gemini / NVIDIA (configurable) |
| Persistence | PostgreSQL + SQLAlchemy |
| Long-term Memory | LangGraph Store (profile context) |
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
- `TAVILY_API_KEY` — required for web search
- Provider API key for the selected provider:
  - `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)
  - `OPENAI_API_KEY` — from [platform.openai.com](https://platform.openai.com)
  - `GEMINI_API_KEY` — from [aistudio.google.com](https://aistudio.google.com)
  - `NVIDIA_API_KEY` — from [build.nvidia.com](https://build.nvidia.com)
- `GOOGLE_CREDENTIALS_FILE` — path to the Google Calendar credentials JSON if calendar tools are enabled

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
I'm a big fan of FC Barca
Search for a football match this weekend
Search for a good restaurant in district 1 in Ho Chi Minh city
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
│   │   ├── nodes.py             # Intent routing, profile memory, web search, tool executor nodes
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
      → Intent classifier routes to chat / todo / web search / profile update
      → Chat responses load profile context from LangGraph store
      → Profile updates store long-term user preferences in LangGraph store
      → Web search can be enriched with profile context for restaurant / sports queries
      → Configured LLM provider reasons + decides tools
      → Executes todo tools (create / list / update / delete / search / remind)
      → Loops until done
      → Returns natural language response
  → Send reply via Telegram
```

## Memory & Search

- Profile memory is stored in LangGraph Postgres store under the `profile` namespace.
- Profile-aware search is applied to underspecified restaurant and sports queries when relevant.
- Explicit user intent always wins over profile hints.

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

## Calendar Tools

Google Calendar MCP tools are loaded read-only for conflict checks and lookups.
Calendar write actions are intentionally disabled; todo creation remains in Postgres.

---