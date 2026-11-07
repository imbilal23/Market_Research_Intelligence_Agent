import secrets
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))


def pytest_addoption(parser):
    parser.addoption("--live-gemini", action="store_true", help="Enable Gemini API smoke tests")


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    # Settings and the SQLAlchemy engine bind their paths during test collection.
    root = session.config._tmp_path_factory.mktemp("app-runtime")
    env = pytest.MonkeyPatch()
    session.config.add_cleanup(env.undo)
    for name, path in {
        "DATA_DIR": root,
        "SQLITE_APP_DB_PATH": root / "app.db",
        "SQLITE_CHECKPOINTER_PATH": root / "checkpoints.db",
        "SQLITE_VECTOR_DB_PATH": root / "vectors.db",
        "UPLOAD_DIR": root / "uploads",
        "TEMP_DIR": root / "tmp",
        "LOGS_DIR": root / "logs",
    }.items():
        env.setenv(name, str(path))
    if not session.config.getoption("--live-gemini"):
        env.setenv("GEMINI_API_KEY", "test_gemini_key")
    env.setenv("JWT_SECRET", secrets.token_urlsafe(48))
    env.setenv("OFFLINE_DEMO", "true")
    env.setenv("OPENAI_API_KEY", "test_key")
    env.setenv("TAVILY_API_KEY", "tvly-dummy-key")
    env.setenv("OPENBLAS_NUM_THREADS", "1")


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as client:
        yield client

