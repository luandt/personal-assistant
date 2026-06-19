from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import model_validator


class Settings(BaseSettings):
    # Telegram
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""

    # LLM
    llm_provider: str = "nvidia"
    llm_model: str = "claude-sonnet-4-20250514"

    nvidia_api_key: str = ""
    nvidia_api_endpoint: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
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

    @model_validator(mode="after")
    def validate_llm_settings(self) -> "Settings":
        provider = (self.llm_provider or "").strip().lower()
        allowed_providers = {"nvidia", "anthropic", "openai", "gemini"}

        if provider not in allowed_providers:
            raise ValueError(
                "Invalid llm_provider. Supported values: nvidia, anthropic, openai, gemini"
            )

        if not (self.llm_model or "").strip():
            raise ValueError("LLM_MODEL must be set and non-empty")

        provider_key_map = {
            "nvidia": self.nvidia_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
        }

        if not (provider_key_map[provider] or "").strip():
            env_key = {
                "nvidia": "NVIDIA_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }[provider]
            raise ValueError(f"Missing API key for provider '{provider}'. Set {env_key}.")

        self.llm_provider = provider
        self.llm_model = self.llm_model.strip()
        return self


    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
