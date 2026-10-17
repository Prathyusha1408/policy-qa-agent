"""Central settings. The API key is read from .env, never hardcoded."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"             # put your policy PDFs here
PERSIST_DIR = BASE_DIR / "chroma_db"     # ChromaDB is saved here
COLLECTION_NAME = "policy_docs"

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4                 # chunks retrieved per query
MAX_QUERY_REWRITES = 1    # retrieval retries before the agent gives up
MAX_HISTORY_MESSAGES = 6  # chat turns passed to the model

NOT_FOUND_MESSAGE = (
    "I couldn't find that in the provided policy documents. "
    "Try rephrasing your question or check that the right documents are indexed."
)


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
        )


def index_exists() -> bool:
    return PERSIST_DIR.is_dir() and any(PERSIST_DIR.iterdir())
