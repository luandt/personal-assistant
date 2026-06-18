"""
APScheduler-based reminder system.
Polls the database every minute for todos that are due and haven't been notified.
"""
import logging
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.db.session import AsyncSessionLocal
from app.db.crud import get_due_reminders, update_todo
from app.telegram.sender import send_message

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def check_and_send_reminders():
    """Check for due todos and send Telegram reminders."""
    now = datetime.now()
    lookahead = now + timedelta(minutes=1)  # fire reminders up to 1 min early

    try:
        async with AsyncSessionLocal() as db:
            todos = await get_due_reminders(db, before=lookahead)

            for todo in todos:
                try:
                    # Get user telegram_id through relationship
                    user = todo.user
                    if not user:
                        continue

                    due_str = todo.due_date.strftime("%b %d at %H:%M") if todo.due_date else "now"
                    msg = (
                        f"⏰ *Reminder*\n\n"
                        f"📌 {todo.title}\n"
                        f"📅 Due: {due_str}\n"
                        f"Priority: {todo.priority.value}"
                    )
                    if todo.description:
                        msg += f"\n📝 {todo.description}"

                    await send_message(user.telegram_id, msg, parse_mode="Markdown")

                    # Mark reminder as sent
                    await update_todo(db, todo_id=todo.id, user_id=todo.user_id, reminder_sent=True)

                except Exception as e:
                    logger.error(f"Failed to send reminder for todo {todo.id}: {e}")

            await db.commit()

    except Exception as e:
        logger.error(f"Reminder check failed: {e}")


def start_scheduler():
    """Start the APScheduler background scheduler."""
    scheduler.add_job(
        check_and_send_reminders,
        trigger=IntervalTrigger(minutes=1),
        id="reminder_checker",
        replace_existing=True,
        max_instances=1,
    )
    # scheduler.start()
    # logger.info("Reminder scheduler started.")


def stop_scheduler():
    """Stop the scheduler gracefully."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        # logger.info("Reminder scheduler stopped.")
