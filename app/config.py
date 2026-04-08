from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_as_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable in a forgiving way."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on", "si"}


@dataclass(slots=True)
class Settings:
    """Centralized application settings."""

    app_name: str = "Asistente de Comercio Ambulatorio"
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    grok_base_url: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local-tfidf")
    chat_model: str = os.getenv("CHAT_MODEL", "grok-4")
    llm_mode: str = os.getenv("LLM_MODE", "mock")
    allow_general_chat: bool = _env_as_bool("ALLOW_GENERAL_CHAT", False)
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.12"))
    memory_history_limit: int = int(os.getenv("MEMORY_HISTORY_LIMIT", "12"))
    memory_max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "40"))
    assistant_max_sources: int = int(os.getenv("ASSISTANT_MAX_SOURCES", "3"))
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    conversations_dir: Path = BASE_DIR / "data" / "processed" / "conversations"
    processed_chunks_file: Path = BASE_DIR / "data" / "processed" / "chunks.json"
    vectorizer_file: Path = BASE_DIR / "data" / "vectorstore" / "tfidf_vectorizer.joblib"
    matrix_file: Path = BASE_DIR / "data" / "vectorstore" / "tfidf_matrix.joblib"


def get_settings() -> Settings:
    """Build and return settings."""

    return Settings()
