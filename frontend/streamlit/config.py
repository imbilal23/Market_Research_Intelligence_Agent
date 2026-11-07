import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LOGS_DIR = BASE_DIR / "logs"
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=LOGS_DIR / "gui_{time:YYYYMMDD}.log",
        level="INFO",
        rotation="00:00",
        retention="7 days",
        compression="zip",
    )
except Exception as log_err:
    sys.stderr.write(f"Warning: Streamlit file logging disabled ({log_err}).\n")


def resolve_backend_url() -> str:
    """Use the service environment, optional local Streamlit secrets, then localhost."""
    url = os.getenv("BACKEND_BASE_URL") or os.getenv("API_BASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("BACKEND_BASE_URL") or st.secrets.get("backend_base_url")
        except (FileNotFoundError, KeyError):
            pass
    # 3. Normalize URL format (ensure /api/v1 path is present)
    if url:
        url = url.strip().rstrip("/")
        if not url.endswith("/api/v1") and "/api/" not in url:
            url = f"{url}/api/v1"
        return url

    # Default fallback for local development
    return "http://127.0.0.1:8000/api/v1"


def resolve_model_names() -> list[str]:
    """Resolve supported intelligence models from Streamlit secrets, env, or defaults."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "MODEL_NAMES" in st.secrets:
            val = st.secrets["MODEL_NAMES"]
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            if isinstance(val, str):
                try:
                    p = json.loads(val)
                    if isinstance(p, list):
                        return [str(x) for x in p]
                except Exception:
                    return [s.strip() for s in val.split(",") if s.strip()]
    except Exception:
        pass

    env_val = os.getenv("MODEL_NAMES")
    if env_val:
        try:
            p = json.loads(env_val)
            if isinstance(p, list):
                return [str(x) for x in p]
        except Exception:
            return [s.strip() for s in env_val.split(",") if s.strip()]

    return ["models/gemini-3.5-flash-lite"]


class Settings(BaseSettings):
    backend_base_url: str = Field(default_factory=resolve_backend_url)
    model_names: list[str] = Field(default_factory=resolve_model_names)

    @field_validator("backend_base_url", mode="before")
    @classmethod
    def clean_backend_url(cls, v: Any) -> str:
        if isinstance(v, str) and v.strip():
            url = v.strip().rstrip("/")
            if not url.endswith("/api/v1") and "/api/" not in url:
                url = f"{url}/api/v1"
            return url
        return resolve_backend_url()

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
        return resolve_model_names()

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="allow",
        populate_by_name=True,
    )


settings = Settings()  # type: ignore
