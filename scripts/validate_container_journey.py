"""Rehearse container replacement with live providers and a dedicated Compose volume."""
import json
import os
from pathlib import Path
import secrets
import subprocess
from urllib.request import urlopen
from uuid import uuid4

from validate_live_journey import port

ROOT = Path(__file__).resolve().parents[1]


def rehearse():
    project = f"research-check-{uuid4().hex[:10]}"
    api_port = port()
    ui_port = port()
    while ui_port == api_port:
        ui_port = port()
    env = dict(os.environ, RESEARCH_DATA_MOUNT="rehearsal-data", PORT="8179",
               BACKEND_HOST_PORT=str(api_port), FRONTEND_HOST_PORT=str(ui_port),
               JOURNEY_PASSWORD=secrets.token_urlsafe(16), JWT_SECRET=secrets.token_urlsafe(48))
    prefix = ["docker", "compose", "-p", project]

    def compose(*args, **kwargs):
        return subprocess.run([*prefix, *args], cwd=ROOT, env=env, check=True, **kwargs)

    def container_id():
        return compose("ps", "-q", "backend", capture_output=True, text=True).stdout.strip()

    def check_health():
        with urlopen(f"http://127.0.0.1:{api_port}/health", timeout=10) as response:
            data = json.load(response)
        assert data["status"] == "healthy", data
        for key in ("database", "vector_search", "checkpointer"):
            assert data[key]["path"].startswith("/data/") and data[key]["exists"], key
        with urlopen(f"http://127.0.0.1:{ui_port}/_stcore/health", timeout=10) as response:
            assert response.status == 200

    def widget_phase(phase):
        compose("exec", "-T", "-e", "JOURNEY_PASSWORD", "-e", "JOURNEY_STREAMLIT_DIR=/app/streamlit",
                "frontend", "python", "-P", "-", phase,
                input=(ROOT / "scripts/validate_live_journey.py").read_bytes())

    try:
        compose("up", "-d", "--build", "--wait", "--wait-timeout", "180")
        check_health()
        widget_phase("upload")
        original = container_id()
        compose("up", "-d", "--force-recreate", "--no-deps", "--wait", "backend")
        replacement = container_id()
        assert original and replacement and replacement != original
        check_health()
        compose("exec", "-T", "backend", "python", "-", input=b"""
import asyncio, sqlite3
from pathlib import Path
from app.ai.rag.vector_store import get_vector_store
with sqlite3.connect('/data/vectors.db') as con:
    thread, user = con.execute('SELECT thread_id, user_id FROM documents LIMIT 1').fetchone()
hits = asyncio.run(get_vector_store().asimilarity_search(
    'retailer enrollment', k=1, filter={'thread_id': thread, 'user_id': user}))
assert hits and '473' in hits[0].page_content
assert any(b'473' in f.read_bytes() for f in Path('/data/uploads').iterdir())
""")
        widget_phase("restart")
        compose("exec", "-T", "backend", "python", "-", input=b"""
import sqlite3, sqlite_vec
from pathlib import Path
assert not list(Path('/data/uploads').iterdir())
with sqlite3.connect('/data/vectors.db') as con:
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    assert con.execute('SELECT count(*) FROM vec_documents').fetchone()[0] == 0
    assert con.execute('SELECT count(*) FROM documents').fetchone()[0] == 0
with sqlite3.connect('/data/app.db') as con:
    assert con.execute('SELECT count(*) FROM documents').fetchone()[0] == 0
""")
        print("Verified Linux readiness, cross-service chat, volume persistence and deletion.")
    except Exception:
        compose("logs", "--tail", "50", "backend", "frontend")
        raise
    finally:
        compose("down", "--volumes", "--rmi", "local")


if __name__ == "__main__":
    rehearse()
