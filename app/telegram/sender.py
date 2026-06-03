"""
Telegram sender — wraps python-telegram-bot's Bot for sending messages.
"""
from telegram import Bot
from telegram.constants import ParseMode
from app.config import get_settings

settings = get_settings()

_bot: Bot = None


def get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=settings.telegram_bot_token)
    return _bot


async def send_message(chat_id: str | int, text: str, parse_mode: str = None) -> None:
    """Send a plain text message to a Telegram chat."""
    bot = get_bot()
    # Telegram max message length is 4096 chars; split if needed
    max_len = 4096
    if len(text) <= max_len:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    else:
        for i in range(0, len(text), max_len):
            await bot.send_message(chat_id=chat_id, text=text[i:i + max_len], parse_mode=parse_mode)


async def send_typing(chat_id: str | int) -> None:
    """Send 'typing...' action."""
    bot = get_bot()
    await bot.send_chat_action(chat_id=chat_id, action="typing")
