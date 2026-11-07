"""Run live source-selection checks with Gemini, Tavily and isolated research data."""
import asyncio
import ast
import os
import secrets
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
REPORT_URL = "https://www.iea.org/reports/global-ev-outlook-2024/executive-summary"
REPORT_NAME = "IEA-Global-EV-Outlook-2024.txt"


async def run():
    from tavily import TavilyClient
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.messages import ToolMessage
    from app.ai.graph.agent import build_retrieval_graph
    from app.ai.rag.vector_store import get_vector_store
    from app.core.config import settings

    if settings.active_provider != "google_genai":
        raise RuntimeError("A real Gemini key is required")

    user, report_thread = str(uuid4()), str(uuid4())

    async def ask(label, question, required, thread):
        graph = build_retrieval_graph(InMemorySaver(), settings.gemini_model)
        result = await asyncio.wait_for(graph.ainvoke(
            {"messages": [("user", question)]},
            {"configurable": {"user_id": user, "thread_id": thread}, "recursion_limit": 12},
        ), timeout=180)
        messages = result["messages"]
        start = max(i for i, m in enumerate(messages) if m.type == "human")
        tools = [m for m in messages[start:] if isinstance(m, ToolMessage)]
        names = [m.name for m in tools]
        answer = messages[-1].content
        if isinstance(answer, list):
            answer = "".join(x if isinstance(x, str) else x.get("text", "") for x in answer)
        print(label, names, answer, sep="\n", flush=True)
        if not required.issubset(names) or (required == {"retrieve_user_documents"} and "tavily" in names):
            raise AssertionError("Unexpected research route")
        if "retrieve_user_documents" in required and REPORT_NAME not in answer:
            raise AssertionError("Missing report attribution")
        if "tavily" in required:
            urls = []
            for m in tools:
                if m.name == "tavily" and str(m.content).startswith("["):
                    urls.extend(row["url"] for row in ast.literal_eval(m.content))
            if not any(url in answer for url in urls):
                raise AssertionError("No citation matches the live search response")

    await ask("current/no uploads", "Search for the latest IEA global electric car sales figure. Give the year and source URL.",
              {"tavily"}, str(uuid4()))
    extracted = TavilyClient(api_key=settings.tavily_api_key).extract(urls=[REPORT_URL])
    raw = extracted["results"][0]["raw_content"]
    if not raw.strip():
        raise RuntimeError("Historical report unavailable")
    store = get_vector_store()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    chunks = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=180).split_text(raw)
    await store.aadd_texts(chunks, metadatas=[{"user_id": user, "thread_id": report_thread,
                                              "file_name": REPORT_NAME} for _ in chunks])
    await ask("historical", "According to the uploaded 2024 IEA report, how many electric cars were sold globally in 2023? Cite the filename.",
              {"retrieve_user_documents"}, report_thread)
    await ask("unrelated upload", "Does my report contain the current list price of the US Nissan Leaf? If missing, look it up and cite the manufacturer.",
              {"retrieve_user_documents", "tavily"}, report_thread)
    await ask("mixed", "Compare the 2023 figure in the uploaded report with the latest actual annual sales in a newer IEA publication. Cite both.",
              {"retrieve_user_documents", "tavily"}, report_thread)


    await ask("cross-check definitions", "Cross-check the uploaded report's global 2023 electric car sales figure against a separate published battery-electric-only estimate for the same year. Explain whether the definitions disagree before comparing totals.",
              {"retrieve_user_documents", "tavily"}, report_thread)

if __name__ == "__main__":
    # The child process owns these temporary storage paths and signing key.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        for key, name in {"DATA_DIR": ".", "SQLITE_APP_DB_PATH": "app.db",
                          "SQLITE_CHECKPOINTER_PATH": "checkpoints.db", "SQLITE_VECTOR_DB_PATH": "vectors.db",
                          "UPLOAD_DIR": "uploads", "TEMP_DIR": "tmp", "LOGS_DIR": "logs"}.items():
            os.environ[key] = str(Path(folder) / name)
        os.environ["JWT_SECRET"] = secrets.token_urlsafe(48)
        asyncio.run(run())
