import asyncio
import json
import threading
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from app.ai.rag import vector_store
from app.schemas import chat
from app.services import document_service
from tests.test_research_access import account


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["index", "search"])
async def test_vector_work_keeps_event_loop_responsive(tmp_path, operation_name):
    started = threading.Event()
    release = threading.Event()

    class SlowEmbeddings(Embeddings):
        def embed_documents(self, texts):
            started.set()
            release.wait(timeout=2)
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            if operation_name == "search":
                started.set()
                release.wait(timeout=2)
            return [1.0, 0.0]

    store = vector_store.SQLiteVectorStore(tmp_path / "vectors.db", SlowEmbeddings())
    work = store.aadd_texts(["test passage"]) if operation_name == "index" else store.asimilarity_search("test passage")
    operation = asyncio.create_task(work)
    try:
        await asyncio.sleep(0.1)
        assert started.is_set()
        assert not operation.done(), "Vector work blocked the event loop until it finished"
    finally:
        release.set()
        await operation


def test_embedding_retry_honors_provider_delay(monkeypatch):
    embedder = Mock()
    embedder.embed_documents.side_effect = [
        RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '43s'}"), [[0.5, 0.5]],
    ]
    monkeypatch.setattr(vector_store, "_make_gemini_embeddings", lambda task: embedder)
    sleep = Mock()
    monkeypatch.setattr(vector_store.time, "sleep", sleep)

    result = vector_store.GeminiDualTaskEmbeddings().embed_documents(["test"])

    assert result == [[0.5, 0.5]]
    sleep.assert_called_once_with(44.0)


def test_daily_quota_stops_without_long_wait(monkeypatch):
    from app.core.provider_errors import provider_error_message

    error = RuntimeError("429 RESOURCE_EXHAUSTED requests per day {'retryDelay': '49012s'}")
    embedder = Mock()
    embedder.embed_documents.side_effect = error
    monkeypatch.setattr(vector_store, "_make_gemini_embeddings", lambda task: embedder)
    sleep = Mock()
    monkeypatch.setattr(vector_store.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="429"):
        vector_store.GeminiDualTaskEmbeddings().embed_documents(["test"])
    sleep.assert_not_called()
    assert "daily usage limit" in provider_error_message(error, "failure")


def test_incomplete_embeddings_do_not_write_partial_document(tmp_path):
    embedder = Mock()
    embedder.embed_query.return_value = [1.0, 0.0]
    embedder.embed_documents.return_value = [[1.0, 0.0]]
    store = vector_store.SQLiteVectorStore(tmp_path / "vectors.db", embedder)

    with pytest.raises(RuntimeError, match="incomplete result"):
        store.add_texts(["first", "second"])
    assert store.similarity_search("") == []


def test_failed_upload_stays_hidden_and_cleans_up(client, monkeypatch):
    from app.core.config import settings
    from app.db.session import async_session

    headers = account(client, "quota-upload")
    thread_id = client.post("/api/v1/threads/", headers=headers).json()["id"]
    before = set(settings.upload_dir.glob("*"))

    async def fail_indexing(file_path, document_id, thread_id, user_id, **kwargs):
        async with async_session() as session:
            assert await document_service.get_documents(thread_id, user_id, session) == []
        raise RuntimeError("429 RESOURCE_EXHAUSTED internal provider diagnostic")

    monkeypatch.setattr(document_service, "index_document_to_vectorstore", fail_indexing)
    response = client.post(f"/api/v1/documents/upload/{thread_id}", headers=headers,
                           files={"file": ("report.txt", b"test data")})

    assert response.status_code == 500
    assert "usage limit" in response.json()["detail"]
    assert "internal provider diagnostic" not in response.text
    assert client.get(f"/api/v1/documents/{thread_id}", headers=headers).json() == []
    assert set(settings.upload_dir.glob("*")) == before


@pytest.mark.asyncio
async def test_stream_sends_heartbeat_and_closes_pending_work(monkeypatch):
    monkeypatch.setattr(chat, "STREAM_HEARTBEAT_SECONDS", 0.01)
    closed = asyncio.Event()

    async def delayed():
        try:
            await asyncio.sleep(60)
            yield "answer"
        finally:
            closed.set()

    stream = chat.safe_stream(delayed())
    assert json.loads(await anext(stream))["type"] == "heartbeat"
    await stream.aclose()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stream_reports_quota_without_diagnostics():
    async def limited():
        raise RuntimeError("429 RESOURCE_EXHAUSTED private diagnostic")
        yield

    events = [json.loads(event) async for event in chat.safe_stream(limited())]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "usage limit" in events[0]["content"]
    assert "private diagnostic" not in events[0]["content"]
