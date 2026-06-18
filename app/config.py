from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Telegram
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""

    # LLM
    nvidia_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    groq_api_key: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/assistant"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/assistant"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # App
    app_env: str = "development"
    secret_key: str = "change_me_in_production"

    # Tavily API key (required for web search)
    tavily_api_key: str = ""

    google_credentials_file: str = "./gcp-oauth.keys.json"
    google_calendar_enabled: bool = True


    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
