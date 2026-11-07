import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Storage and logs directory defaults
DATA_DIR_DEFAULT = Path(os.getenv("DATA_DIR", os.getenv("STORAGE_DIR", str(BASE_DIR / "backend" / "data"))))
LOGS_DIR_DEFAULT = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
TEMP_DIR_DEFAULT = Path(os.getenv("TEMP_DIR", str(BASE_DIR / "tmp")))

# Safe directory creation with fallback
try:
    LOGS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=LOGS_DIR_DEFAULT / "api_{time:YYYYMMDD}.log",
        level="INFO",
        rotation="00:00",
        retention="7 days",
        compression="zip",
    )
except Exception as log_err:
    # Graceful fallback if container root is read-only
    sys.stderr.write(f"Warning: File logging disabled ({log_err}). Logging to stderr.\n")

logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    offline_demo: bool = False

    # Server Binding
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # --- Gemini / Google AI Studio ---
    gemini_api_key: SecretStr = Field(default=SecretStr(""), alias="GEMINI_API_KEY")
    # LLM model: configurable via GEMINI_MODEL env var. Default is models/gemini-3.5-flash-lite (stable).
    # User-supplied value (e.g. models/gemini-3.5-flash-lite) is used when set.
    gemini_model: str = Field(default="models/gemini-3.5-flash-lite", alias="GEMINI_MODEL")
    # Embedding model: gemini-embedding-2 supports output_dimensionality and task_type
    embedding_model: str = Field(default="models/gemini-embedding-2", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")

    # --- Legacy OpenAI fallback (kept for compatibility; not used when GEMINI_API_KEY is set) ---
    api_key: SecretStr = Field(default=SecretStr("test_key"), alias="OPENAI_API_KEY")
    tavily_api_key: str = Field(default="test_tavily_key")
    model_provider: str = Field(default="openai")
    model_names: list[str] = Field(default_factory=lambda: ["models/gemini-3.5-flash-lite"])
    model_base_url: str | None = None
    embeddings_model_name: str = Field(default="text-embedding-3-large")
    embeddings_base_url: str | None = None

    # Auth & Tokens
    token_bearer_url: str = Field(default="/api/v1/auth/login")
    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = Field(default="HS256")
    access_token_expiry_mins: int = Field(default=1500)
    refresh_token_expiry_days: int = Field(default=1)

    # Storage & Persistence (Railway volume mount support: e.g. /data)
    data_dir: Path = Field(default_factory=lambda: DATA_DIR_DEFAULT)
    sqlite_app_db_path: Path = Field(default_factory=lambda: DATA_DIR_DEFAULT / "app.db")
    sqlite_checkpointer_path: Path = Field(default_factory=lambda: DATA_DIR_DEFAULT / "checkpoints.db")
    sqlite_vector_db_path: Path = Field(default_factory=lambda: DATA_DIR_DEFAULT / "vectors.db")
    upload_dir: Path = Field(default_factory=lambda: DATA_DIR_DEFAULT / "uploads")
    temp_dir: Path = Field(default_factory=lambda: TEMP_DIR_DEFAULT)

    # CORS configuration
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("jwt_secret")
    @classmethod
    def require_private_signing_key(cls, value: str) -> str:
        # Reject signing keys previously distributed in source or environment examples.
        if hashlib.sha256(value.encode()).hexdigest() in {
            "8f8f76fb8591d0501127f15eecfe7c02616f4ae2106d58161d6990678088d95e",
            "0770823b2a9b86c8ca3376e3ce26233e58248cc6b8e9c6926bc827851fe256b1",
        } or len(value.strip()) < 32:
            raise ValueError("Configure a private random JWT_SECRET of at least 32 characters")
        return value

    @field_validator("model_names", mode="before")
    @classmethod
    def parse_model_names(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(item) for item in v]
        return ["models/gemini-3.5-flash-lite"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(item) for item in v]
        return ["*"]

    @property
    def database_uri(self) -> str:
        """Generate SQLite async connection string for SQLAlchemy."""
        path = Path(self.sqlite_app_db_path).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create database directory {path.parent}: {e}")
        posix = path.as_posix()
        if posix.startswith("/"):
            return f"sqlite+aiosqlite:///{posix}"
        return f"sqlite+aiosqlite:///{posix}"

    @property
    def checkpointer_uri(self) -> str:
        """Generate SQLite connection string for LangGraph checkpointer."""
        path = Path(self.sqlite_checkpointer_path).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create checkpointer directory {path.parent}: {e}")
        return str(path)

    @property
    def vector_db_path(self) -> str:
        """Generate SQLite vector store file path."""
        path = Path(self.sqlite_vector_db_path).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create vector database directory {path.parent}: {e}")
        return str(path)

    @property
    def gemini_api_key_str(self) -> str:
        """Return the raw GEMINI_API_KEY string for safe checks without exposing the value in logs."""
        return (
            self.gemini_api_key.get_secret_value()
            if hasattr(self.gemini_api_key, "get_secret_value")
            else str(self.gemini_api_key)
        )

    @property
    def active_provider(self) -> str:
        """
        Determine the active LLM/embedding provider.
        - Returns 'google_genai' when GEMINI_API_KEY is set and non-empty.
        - Returns 'openai' when OPENAI_API_KEY is set (legacy path).
        - Returns 'mock' when no valid key is available.
        """
        key = self.gemini_api_key_str
        if key and key not in ("", "test_gemini_key"):
            return "google_genai"
        openai_key = (
            self.api_key.get_secret_value()
            if hasattr(self.api_key, "get_secret_value")
            else str(self.api_key)
        )
        if openai_key and openai_key not in ("", "test_key"):
            return "openai"
        return "mock"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        hide_input_in_errors=True,
        extra="allow",
        populate_by_name=True,
    )


settings = Settings()  # type: ignore

# Backward-compatibility directory aliases
DATA_DIR = settings.data_dir
TEMP_DIR = settings.temp_dir
LOGS_DIR = LOGS_DIR_DEFAULT

# Ensure runtime directories exist
try:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
except Exception as dir_err:
    logger.warning(f"Could not ensure initial storage directories: {dir_err}")


