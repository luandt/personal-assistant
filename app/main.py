"""
FastAPI application entrypoint.
Handles startup (DB init, graph build, scheduler) and the Telegram webhook.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.session import init_db, AsyncSessionLocal
from app.agent.graph import build_graph
from app.agent.store import generate_store
from app.telegram.webhook import router as webhook_router
from app.scheduler.reminders import start_scheduler, stop_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("Starting up Personal Assistant...")

    # 1. Init database tables
    try:
        await init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.warning("Continuing startup without database. Some features may be unavailable.")

    # 2. Build LangGraph agent (with tools + checkpointer + custom store)
    store_cm = generate_store()
    store = None
    try:
        # Enter the store context and keep it open for the app lifetime
        store = await store_cm.__aenter__()
        await build_graph(db_session_factory=AsyncSessionLocal, store=store)
        logger.info("LangGraph agent built with custom store.")
    except Exception as e:
        logger.error(f"Failed to build LangGraph agent: {e}")
        logger.warning("Continuing startup. Agent features will be unavailable.")

    # 3. Register Telegram webhook
    if settings.telegram_bot_token and settings.telegram_webhook_url:
        try:
            from app.telegram.sender import get_bot
            bot = get_bot()
            webhook_url = settings.telegram_webhook_url
            await bot.set_webhook(url=webhook_url)
            logger.info(f"Telegram webhook registered: {webhook_url}")
        except Exception as e:
            logger.warning(f"Could not set Telegram webhook: {e}")

    # 4. Start reminder scheduler
    try:
        start_scheduler()
        logger.info("Reminder scheduler started.")
    except Exception as e:
        logger.warning(f"Failed to start reminder scheduler: {e}")


    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        # Close the store context if it was opened
        if store is not None:
            await store_cm.__aexit__(None, None, None)
            logger.info("Custom store shut down.")
    except Exception as e:
        logger.warning(f"Failed to shut down custom store: {e}")
    stop_scheduler()


app = FastAPI(
    title="Personal Assistant",
    description="Telegram-based personal assistant with AI-powered todo management",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(webhook_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "personal-assistant"}


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})
