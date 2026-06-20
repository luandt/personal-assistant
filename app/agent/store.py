import contextlib

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.store.base import IndexConfig
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.config import get_settings

settings = get_settings()

# Initialize embeddings service (adjust model as needed via env/settings)
embeddings = NVIDIAEmbeddings(model="nv-embedqa-e5-v5")

@contextlib.asynccontextmanager
async def generate_store():
    """Async context manager that yields an AsyncPostgresStore configured for semantic search.

    This will be discovered by LangGraph/LangSmith when you point `langgraph.json`'s
    `store.path` to `app.agent.store:generate_store`.
    """
    async with AsyncPostgresStore.from_conn_string(
        settings.database_url_sync,
        index=IndexConfig(dims=1536, embed=embeddings, fields=["user_memory"]),
    ) as store:
        # Ensure tables/indexes exist on first run
        await store.setup()
        yield store
