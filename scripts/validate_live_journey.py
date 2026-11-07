"""Exercise the Streamlit widgets against a real API and restart with isolated storage."""
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
FRONT = Path(os.environ.get("JOURNEY_STREAMLIT_DIR", ROOT / "frontend" / "streamlit"))
NAME = "analyst-survey.txt"
FIGURE = "473"


def interface(phase):
    from streamlit.testing.v1 import AppTest

    sys.path.insert(0, str(FRONT))
    app = AppTest.from_file(str(FRONT / "main.py"), default_timeout=180).run()

    def click(label):
        next(b for b in app.button if b.label == label).click().run()

    def fill(label, value):
        next(w for w in app.text_input if w.label == label).set_value(value)

    email = "step5@example.com"
    password = os.environ["JOURNEY_PASSWORD"]
    if phase == "upload":
        click("Register")
        fill("Corporate Email *", email)
        fill("Username *", "analyst5")
        fill("Password *", password)
        click("Register Analyst")
    else:
        click("Sign In")
    fill("Corporate Email *", email)
    fill("Password *", password)
    click("Sign In")
    assert app.session_state["user"].is_authenticated
    if phase == "upload":
        app.file_uploader[0].upload(NAME, b"Analyst survey. The pilot enrolled 473 retailers.", "text/plain").run()
        app.button(key="btn_ingest_docs").click().run()
    else:
        next(b for b in app.button if str(b.key).startswith("select_")).click().run()
    thread = app.session_state["thread"]
    assert [d["file_name"] for d in thread.documents] == [NAME]
    if phase == "restart":
        assert any(m["role"] == "ai" and FIGURE in m["content"] and NAME in m["content"]
                   for m in thread.messages), "Checkpoint history did not reopen"
    next(w for w in app.checkbox if "Web" in w.label).uncheck().run()
    question = ("How many retailers enrolled in the pilot? Cite the uploaded filename and answer briefly."
                if phase == "upload" else "What was twice the enrollment we just discussed? Give the number.")
    app.chat_input(key="market_research_chat_input").set_value(question).run()
    thread = app.session_state["thread"]
    assert not app.exception
    assert thread.messages[-1]["role"] == "ai", "No completed streamed answer"
    answer = thread.messages[-1]["content"]
    assert (FIGURE if phase == "upload" else "946") in answer, answer[:400]
    assert "'signature':" not in answer
    if phase == "upload":
        assert NAME in answer
    else:
        app.button(key="kb_del_" + str(thread.documents[0]["id"])).click().run()
        assert not app.session_state["thread"].documents
    print("Verified Streamlit phase:", phase, flush=True)


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url, process):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("Service exited during startup")
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def journey():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        data = Path(folder)
        api_port, ui_port = port(), port()
        env = dict(os.environ, HOST="127.0.0.1", PORT=str(api_port),
                   BACKEND_BASE_URL=f"http://127.0.0.1:{api_port}/api/v1",
                   JOURNEY_PASSWORD=secrets.token_urlsafe(14), JWT_SECRET=secrets.token_urlsafe(48),
                   PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", DATA_DIR=folder)
        for key, filename in {"SQLITE_APP_DB_PATH":"app.db", "SQLITE_CHECKPOINTER_PATH":"checkpoints.db",
                              "SQLITE_VECTOR_DB_PATH":"vectors.db", "UPLOAD_DIR":"uploads",
                              "LOGS_DIR":"logs", "TEMP_DIR":"tmp"}.items():
            env[key] = str(data / filename)
        for phase in ("upload", "restart"):
            processes = []
            try:
                for args in (["scripts/run_backend.py"], ["-m", "streamlit", "run", str(FRONT / "main.py"),
                             "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={ui_port}"]):
                    processes.append(subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env,
                                                      stdout=subprocess.DEVNULL, stderr=None))
                wait_ready(f"http://127.0.0.1:{api_port}/health", processes[0])
                wait_ready(f"http://127.0.0.1:{ui_port}/_stcore/health", processes[1])
                subprocess.run([sys.executable, __file__, phase], cwd=ROOT, env=env, check=True, timeout=420)
            finally:
                for process in processes:
                    if os.name == "nt" and process.poll() is None:
                        # Windows venv launchers can own a child server process.
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                       check=True, stdout=subprocess.DEVNULL)
                    else:
                        process.terminate()
                    process.wait(timeout=20)
            for service_port in (api_port, ui_port):
                with socket.socket() as check:
                    assert check.connect_ex(("127.0.0.1", service_port)) != 0, "Old service still running"
            expected = phase == "upload"
            assert bool(list((data / "uploads").glob("*"))) == expected
            for db, table in (("app.db", "documents"), ("vectors.db", "documents"), ("vectors.db", "vec_documents")):
                import sqlite_vec
                with sqlite3.connect(data / db) as conn:
                    if db == "vectors.db":
                        conn.enable_load_extension(True)
                        sqlite_vec.load(conn)
                    assert bool(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) == expected
                conn.close()
        print("Verified restart, follow-up, and file, metadata and vector cleanup.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        interface(sys.argv[1])
    else:
        journey()
