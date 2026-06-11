"""
Telegram webhook handler.
Receives updates from Telegram and routes them to the LangGraph agent.
"""
import logging

from pymupdf import message
from fastapi import APIRouter, Request, HTTPException, Depends
from telegram import Update
from telegram.ext import Application

from app.config import get_settings
from app.db.session import get_db, AsyncSessionLocal
from app.db.crud import get_or_create_user
from app.telegram.sender import send_message, send_typing
from app.agent.graph import get_graph, run_agent

import httpx
import tempfile
import os
from app.telegram.sender import get_bot

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/webhook-info")
async def webhook_info():
    """Check webhook registration status and configuration."""
    try:
        bot = get_bot()
        
        if not settings.telegram_bot_token:
            return {"error": "telegram_bot_token not configured"}
        if not settings.telegram_webhook_url:
            return {"error": "telegram_webhook_url not configured"}
        
        webhook_info = await bot.get_webhook_info()
        return {
            "status": "ok",
            "bot_token_configured": bool(settings.telegram_bot_token),
            "webhook_url_configured": settings.telegram_webhook_url,
            "webhook_info": {
                "url": webhook_info.url,
                "has_custom_certificate": webhook_info.has_custom_certificate,
                "pending_update_count": webhook_info.pending_update_count,
                "last_error_date": webhook_info.last_error_date,
                "last_error_message": webhook_info.last_error_message,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get webhook info: {e}", exc_info=True)
        return {"error": str(e)}


@router.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Receive Telegram webhook updates.
    Telegram sends POST requests here for every update.
    """
    # Validate secret token (optional but recommended)
    data = await request.json()
    logger.info(f"Received webhook update: {data}")

    try:
        update = Update.de_json(data, None)
    except Exception as e:
        logger.error(f"Failed to parse Telegram update: {e}")
        raise HTTPException(status_code=400, detail="Invalid update")

    # Only handle text messages
    # if not update.message or not update.message.text:
    #     return {"ok": True}
    if not update.message or (not update.message.text and not update.message.voice):
        return {"ok": True}

    message = update.message
    telegram_user = message.from_user
    chat_id = message.chat_id

    if message.voice:
        try:
            await send_typing(chat_id)
            file_path = await download_voice(message.voice.file_id)
            text = await transcribe_voice(file_path)
            os.unlink(file_path)  # clean up temp file
            logger.info(f"Transcribed voice: {text}")
        except Exception as e:
            logger.error(f"Voice transcription failed: {e}")
            await send_message(chat_id, "Sorry, I couldn't understand your voice message.")
            return {"ok": True}

    # Handle text message
    elif message.text:
        text = message.text.strip()
    else:
        return {"ok": True}
        
    # Ignore commands for now (can extend later)
    if text.startswith("/start"):
        logger.info(f"Received /start from user {telegram_user.id} (chat_id: {chat_id})")
        try:
            await send_message(chat_id, "👋 Hi! I'm your personal assistant. Tell me what to add to your todo list, or ask me anything!")
            logger.info(f"Successfully sent /start response to chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send /start message to chat_id {chat_id}: {e}", exc_info=True)
        return {"ok": True}

    if text.startswith("/help"):
        logger.info(f"Received /help from user {telegram_user.id} (chat_id: {chat_id})")
        help_text = (
            "🤖 *Personal Assistant*\n\n"
            "I understand natural language! Try:\n"
            "• *'Add buy groceries tomorrow'*\n"
            "• *'What do I have this week?'*\n"
            "• *'Mark gym as done'*\n"
            "• *'Remind me to call mom tomorrow 3pm'*\n"
            "• *'Delete everything tagged #work'*\n"
            "• *'Search for gym'*"
        )
        try:
            await send_message(chat_id, help_text, parse_mode="Markdown")
            logger.info(f"Successfully sent /help response to chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send /help message to chat_id {chat_id}: {e}", exc_info=True)
        return {"ok": True}

    # Show typing indicator
    try:
        await send_typing(chat_id)
    except Exception as e:
        logger.warning(f"Failed to send typing indicator: {e}")

    # Get or create user and run agent
    try:
        async with AsyncSessionLocal() as db:
            user = await get_or_create_user(
                db,
                telegram_id=str(telegram_user.id),
                username=telegram_user.username,
                first_name=telegram_user.first_name,
            )
            await db.commit()
            user_id = user.id

        # Run agent
        try:
            graph = await get_graph()
            response = await run_agent(
                graph=graph,
                user_id=user_id,
                telegram_id=str(telegram_user.id),
                chat_id=str(chat_id),
                user_message=text,
            )
            await send_message(chat_id, response)
        except Exception as e:
            logger.exception(f"Agent error for user {telegram_user.id}: {e}")
            await send_message(chat_id, "Sorry, something went wrong. Please try again.")
    except Exception as e:
        logger.error(f"Database/agent error for user {telegram_user.id}: {e}", exc_info=True)
        await send_message(chat_id, "Sorry, I'm having trouble accessing my memory right now. Please try again later.")

    return {"ok": True}

async def transcribe_voice(file_path: str) -> str:
    """Transcribe using Groq Whisper API (free tier)."""
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                data={  "model": "whisper-large-v3",
                        "language": "en",
                        "prompt": "todo tasks, reminders, appointments",},
                files={"file": ("voice.ogg", f, "audio/ogg")},
            )
        response.raise_for_status()
        return response.json()["text"]


async def download_voice(file_id: str) -> str:
    """Download voice file from Telegram and return ldocal path."""
    
    
    bot = get_bot()
    
    # Get file path from Telegram
    file = await bot.get_file(file_id)
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await file.download_to_memory(tmp)
        return tmp.name