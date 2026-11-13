# 🤖 Market Research Intelligence Agent

An agentic Retrieval-Augmented Generation (RAG) system built with **FastAPI** and **LangGraph**, featuring streaming responses, local vector storage powered by **SQLite + sqlite-vec**, and an interactive **Streamlit** UI. The system supports user authentication, threaded conversations with persistent memory via LangGraph SQLite checkpointers, and tool-augmented reasoning (document retrieval + web search).

## 🚀 Features

- **Agentic RAG with LangGraph**: ReAct-style agent with tools for document retrieval and web search
- **Streaming responses end-to-end**: Real-time token streaming from backend to the Streamlit UI
- **Threaded conversations**: Per-user threads with persistent histories stored via SQLite checkpointers
- **SQLite + sqlite-vec**: Local, zero-configuration vector storage and fast similarity retrieval over uploaded documents
- **Authentication and JWT**: Signup, login, refresh; per-user isolation for threads and docs
- **Document ingestion**: PDF, DOCX, and TXT support with chunking and async indexing
- **Tooling**: Built-in `retrieve_user_documents` and Tavily web search integration
- **Async-first backend**: FastAPI + SQLAlchemy 2.0 async, production-ready logging and healthchecks

## 💻 Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain, SQLAlchemy, Pydantic v2
- **Persistence & Metadata**: SQLite (`app.db` via `aiosqlite`)
- **Vector Store**: SQLite + `sqlite-vec` virtual tables (`vectors.db`)
- **Checkpointer**: LangGraph SQLite Checkpointer (`checkpoints.db` via `langgraph-checkpoint-sqlite`)
- **Frontend**: Streamlit
- **LLM/Embeddings**: Google AI Studio Gemini for generation and embeddings

## 📋 Prerequisites

- Python 3.12+
- Docker and Docker Compose (optional for containerized deployment)

## 📦 Quick Start (Docker Compose)

1. On a fresh checkout, copy the template and set your Gemini key, Tavily key and private JWT signing key. Keep your existing .env on an initialized checkout.
   ```bash
   cp .env.example .env
   ```
2. Start the full stack:
   ```bash
   docker compose up --build
   ```

Services:
- Backend API: `http://localhost:8000/api/v1`
- API Docs: `http://localhost:8000/api/v1/docs`
- Frontend UI: `http://localhost:8501`

### Local container rehearsal

From this directory with the project virtual environment active, run:

```bash
python scripts/validate_container_journey.py
```

This live check requires Docker Compose and real provider credentials. It builds both images, starts an isolated project on loopback ports, and runs the Streamlit widget journey from inside the frontend container. It recreates the backend on the same named volume, checks health on a nondefault internal port, reopens the thread, asks a follow-up and verifies document deletion. Its containers and temporary volume are removed afterwards.

Verified locally on 7 November 2026 with Linux containers, Docker Engine 29.8.2, Compose 5.5.1 and Python 3.12.11. The rehearsal passed upload, sourced chat, backend replacement, vector lookup after replacement, history recovery and deletion. Application dependency manifests use version ranges, so rerun the rehearsal after rebuilding with newer dependencies.

Normal Compose runs map `./backend/data` to `/data`. Keep that directory for the application database, checkpoint database, vector index and uploads. Both Dockerfiles use the application root as their build context. The backend image reads `PORT`, defaults `DATA_DIR` to `/data`, and serves readiness at `/health`.

## Railway hosting

The deployment uses one Railway project and environment. The browser connects to the public Streamlit domain over HTTPS. Streamlit makes server-side HTTP requests to FastAPI using Railway private DNS. Gemini, Tavily and SQLite belong to the backend.

### GitHub-connected setup

Connect both services to the same GitHub repository and branch. Set each service's Root Directory to the repository root (`/`, the default). Set `RAILWAY_DOCKERFILE_PATH` separately on each service and configure its health check and restart policy in Railway's service settings. Both Dockerfiles copy files using the repository-root build context. Leave Config File Path empty. [Railway Dockerfile paths](https://docs.railway.com/builds/dockerfiles#custom-dockerfile-path)

The current services are `Market_Research_Intelligence_Agent` and `frontend`. For a fresh installation, the examples below name the backend service `backend`. Match cross-service variable references to the actual backend service name in your project.

| Service setting | backend | frontend |
|---|---|---|
| `RAILWAY_DOCKERFILE_PATH` service variable | `backend/Dockerfile` | `frontend/Dockerfile` |
| Docker image startup (leave Railway start override empty) | `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}` | `streamlit run streamlit/main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}` |
| Service variable | `PORT=8000` | `PORT=8501` |
| Health check | `/health` checks SQLite, checkpoints and sqlite-vec | `/_stcore/health` checks Streamlit |
| Restart policy | On failure, maximum 10 retries | On failure, maximum 10 retries |
| Networking | Private network only. Leave public domains and TCP proxies empty. | Generate a public HTTPS domain targeting port 8501. |
| Storage | Attach one persistent volume at `/data`. Use one replica. | Ephemeral filesystem. |

**Backend variables.** Set private values for `GEMINI_API_KEY`, `TAVILY_API_KEY`, and a stable random `JWT_SECRET` of at least 32 characters. Set `GEMINI_MODEL=models/gemini-3.5-flash-lite`, `EMBEDDING_MODEL=models/gemini-embedding-2`, `EMBEDDING_DIMENSIONS=768`, and `DATA_DIR=/data`. With that directory the defaults are `/data/app.db`, `/data/checkpoints.db`, `/data/vectors.db`, and `/data/uploads`. Individual overrides are `SQLITE_APP_DB_PATH`, `SQLITE_CHECKPOINTER_PATH`, `SQLITE_VECTOR_DB_PATH`, and `UPLOAD_DIR`. Keep all four on the mounted volume. Set provider and signing secrets only on this service. [Railway volumes](https://docs.railway.com/volumes)

**Frontend variables.** In Railway set `BACKEND_BASE_URL=http://${{backend.RAILWAY_PRIVATE_DOMAIN}}:${{backend.PORT}}/api/v1` and `MODEL_NAMES=["models/gemini-3.5-flash-lite"]`. The explicit backend PORT above supplies the value for this cross-service reference. Private HTTP is server-to-server. For this new environment the IPv4 listener on 0.0.0.0 uses Railway's private A record. [Railway private domains](https://docs.railway.com/networking/domains/working-with-domains)

After deployment, check the frontend domain, reach backend health from the frontend service, then verify upload, retrieval and history after backend replacement on its volume. Local Compose exposes the API on loopback for development.
## Uploading reports

Choose a PDF, Word document or text file, then select **Add files to chat**. Large reports can take several minutes while Gemini prepares the searchable passages. The app allows up to 30 minutes for an upload response and displays the file in the chat after indexing succeeds. If a timeout occurs, reopen the chat and check its files before uploading another copy.

Indexing runs in a worker thread so other API requests can continue. Gemini retries respect the provider's suggested delay, with bounded attempts. Chat streams send keep-alive events during long searches or model responses. Usage-limit errors identify the provider limit in the UI. Actual throughput depends on the Gemini project's quota. [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

For an assessment restricted to an uploaded report, state that restriction in your question. You can also turn off **Search the web**. Report evidence includes filenames and PDF page numbers for checking the answer.

## Local development

Run these commands from the project root. Create the environment once on a fresh checkout, then follow the configuration instructions above.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the backend and frontend in separate terminals:

```powershell
.\.venv\Scripts\python.exe scripts/run_backend.py
.\.venv\Scripts\python.exe scripts/run_frontend.py
```

On Linux or macOS, use `.venv/bin/python` in place of the Windows interpreter. The root dependency manifest includes both services and pytest. Each service retains its own manifest for independent builds.

## Project guide

| Location | Purpose |
|---|---|
| `backend/app/` | FastAPI routes, services, auth, SQLite storage, LangGraph orchestration and research tools |
| `frontend/streamlit/` | Streamlit entry point, session state, components and API client |
| `frontend/.streamlit/` | Streamlit server configuration and optional local settings example for the frontend |
| `scripts/` | Local launchers, reindexing, live research checks and container validation |
| `tests/` | Offline regression and adaptive retrieval tests, plus opt-in Gemini checks |

Run the full offline suite and package checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pip check
```

For real provider verification, run `python -m pytest tests/test_gemini_integration.py::TestGeminiLiveSmoke --live-gemini` in the active environment. Live and container checks require configured credentials and consume provider quota.
## 🔧 Environment Variables

Create a project-root `.env` (both backend and frontend read from it). Key settings:

Research provider settings:
- `GEMINI_API_KEY` and `GEMINI_MODEL` select the generation provider and model.
- `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` select the embedding model and vector width (768 by default).
- `MODEL_NAMES` configures the Streamlit model picker, e.g. `["models/gemini-3.5-flash-lite"]`.
- `TAVILY_API_KEY` enables live web search.
Auth and tokens:
- `TOKEN_BEARER_URL` (default `/api/v1/auth/login`)
- `JWT_SECRET` (required; generate a private value with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Replacing this key requires users to sign in again.)
- `OFFLINE_DEMO=true` explicitly enables simulated chat when no model credential is configured. Leave this unset for normal research.
- `JWT_ALGORITHM` (e.g., `HS256`)
- `ACCESS_TOKEN_EXPIRY_MINS` (e.g., `1440`)
- `REFRESH_TOKEN_EXPIRY_DAYS` (e.g., `1`)

SQLite Database Paths:
- `SQLITE_APP_DB_PATH` (default `backend/data/app.db`)
- `SQLITE_CHECKPOINTER_PATH` (default `backend/data/checkpoints.db`)
- `SQLITE_VECTOR_DB_PATH` (default `backend/data/vectors.db`)

Frontend:
- `BACKEND_BASE_URL` (e.g., `http://127.0.0.1:8000/api/v1` when running locally)

Example values are provided in `.env.example`.

## 🧩 API Overview

Base URL: `/api/v1`

Auth:
- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/logout`
- `GET /auth/refresh-token`

Users:
- `GET /users/me`
- `PUT /users/user-profile/{user_id}`
- `DELETE /users/user-profile/{user_id}`

Threads:
- `POST /threads/` (create)
- `GET /threads/` (list)
- `GET /threads/{thread_id}` (get)
- `PATCH /threads/{thread_id}` (update title)
- `DELETE /threads/{thread_id}` (delete + cascade cleanup of memory and vectors)

Documents:
- `GET /documents/{thread_id}` (list)
- `POST /documents/upload/{thread_id}` (upload + async index)
- `DELETE /documents/{document_id}` (remove + delete chunks from sqlite-vec)

Chat and streaming:
- `POST /chat/` (guest streaming chat; accepts the last eight messages for follow-up questions)
- `POST /chat/{thread_id}` (signed-in chat with saved history; use_documents and use_web select the available tools). The frontend automatically enables file search when the current chat has uploaded files. Search options contains the web-search toggle.
- `GET /chat/{thread_id}` (retrieve persisted chat history)

API docs:
- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

## 📡 Streaming Protocol

Both chat endpoints stream newline-delimited JSON events. Event types include:
- `llm_chunk`: incremental model output
- `tool_call`: tool name and arguments when the agent invokes a tool
- `tool_result`: tool output returned to the agent
- `error`: terminal interruption; discard the incomplete answer and allow retry

## 🔄 Architecture

1. **Ingestion & Indexing**
   - PDF, DOCX, TXT loaders; chunking via `RecursiveCharacterTextSplitter`
   - Async indexing into `sqlite-vec` using virtual table `vec_documents USING vec0`

2. **Retrieval**
   - Semantic similarity search filtered by `thread_id` and `user_id`
   - Tool: `retrieve_user_documents` leverages SQLiteVectorStore retriever

3. **Agent & Generation**
   - LangGraph ReAct agent (`create_react_agent`) with tools (documents + Tavily)
   - Configurable models via `MODEL_NAMES`
   - End-to-end streaming

4. **Memory & Persistence**
   - LangGraph SQLite checkpointer (`AsyncSqliteSaver`) stores per-thread chat histories in `checkpoints.db`
   - Thread deletion cleans up checkpointer state and related vector chunks

## 📝 License

Licensed under the [MIT License](./LICENSE).

