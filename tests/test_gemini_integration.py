"""
tests/test_gemini_integration.py
=================================
Tests for Google Gemini LLM and embedding integration.
All tests mock Gemini API calls for offline/CI execution.
If GEMINI_API_KEY is set in the environment, a lightweight live smoke-test
is also executed.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ─── Helper: clear the global embedding singleton between tests ───────────────

def _reset_embedding_singleton():
    import app.ai.rag.vector_store as vs_module
    vs_module._embeddings = None


def _reset_vector_store_singleton():
    import app.ai.rag.vector_store as vs_module
    vs_module._vector_store_instance = None


# ─── 1. Configuration ───────────────────────────────────────────────────────

class TestGeminiConfig:
    def test_active_provider_is_mock_without_key(self, monkeypatch):
        """Without any API key, active_provider should be 'mock'."""
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        from importlib import reload
        import app.core.config as cfg_mod
        # Re-read settings with patched env
        from app.core.config import Settings
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.active_provider == "mock"

    def test_active_provider_with_gemini_key(self, monkeypatch):
        """With a non-empty GEMINI_API_KEY, active_provider is 'google_genai'."""
        from app.core.config import Settings
        from pydantic import SecretStr
        s = Settings(_env_file=None, GEMINI_API_KEY=SecretStr("AIza-test-key"))  # type: ignore[call-arg]
        assert s.active_provider == "google_genai"

    def test_embedding_dimensions_configurable(self, monkeypatch):
        """EMBEDDING_DIMENSIONS env var should be surfaced in settings."""
        from app.core.config import Settings
        s = Settings(_env_file=None, EMBEDDING_DIMENSIONS=512)  # type: ignore[call-arg]
        assert s.embedding_dimensions == 512

    def test_gemini_model_default(self):
        from app.core.config import Settings
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert "gemini" in s.gemini_model.lower()

    def test_embedding_model_default(self):
        from app.core.config import Settings
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert "gemini" in s.embedding_model.lower()


# ─── 2. Embedding routing ───────────────────────────────────────────────────

class TestEmbeddingRouting:
    def setup_method(self):
        _reset_embedding_singleton()

    def teardown_method(self):
        _reset_embedding_singleton()

    def test_returns_fake_when_no_key(self, monkeypatch):
        """No GEMINI_API_KEY → DeterministicFakeEmbedding returned."""
        monkeypatch.setattr("app.core.config.settings.gemini_api_key",
                            __import__("pydantic").SecretStr(""))
        from app.ai.rag.vector_store import get_embeddings
        emb = get_embeddings()
        assert isinstance(emb, DeterministicFakeEmbedding)

    def test_fake_embedding_uses_configured_dimensions(self, monkeypatch):
        """Offline fake embedding uses settings.embedding_dimensions, not hardcoded 1536."""
        monkeypatch.setattr("app.core.config.settings.gemini_api_key",
                            __import__("pydantic").SecretStr(""))
        monkeypatch.setattr("app.core.config.settings.embedding_dimensions", 768)
        _reset_embedding_singleton()
        from app.ai.rag.vector_store import get_embeddings
        emb = get_embeddings()
        assert isinstance(emb, DeterministicFakeEmbedding)
        vec = emb.embed_query("test")
        assert len(vec) == 768

    def test_gemini_dual_task_routes_correctly(self, monkeypatch):
        """GeminiDualTaskEmbeddings calls doc embedder for embed_documents and query embedder for embed_query."""
        fake_doc_emb = [[0.1] * 768]
        fake_query_emb = [[0.2] * 768]

        mock_doc = MagicMock()
        mock_doc.embed_documents.return_value = fake_doc_emb
        mock_query = MagicMock()
        mock_query.embed_documents.return_value = fake_query_emb

        with patch("app.ai.rag.vector_store._make_gemini_embeddings") as mock_make:
            mock_make.side_effect = [mock_doc, mock_query]
            from app.ai.rag.vector_store import GeminiDualTaskEmbeddings
            dual = GeminiDualTaskEmbeddings()

        result_docs = dual.embed_documents(["hello world"])
        assert result_docs == fake_doc_emb
        mock_doc.embed_documents.assert_called_once_with(["hello world"])

        result_query = dual.embed_query("search query")
        assert result_query == fake_query_emb[0]
        mock_query.embed_documents.assert_called_once_with(["search query"])

    def test_gemini_embeddings_batch_processing(self, monkeypatch):
        """embed_documents sends chunks in batches of BATCH_SIZE."""
        def fake_embed(texts):
            return [[0.5] * 768 for _ in texts]

        with patch("app.ai.rag.vector_store._make_gemini_embeddings") as mock_make:
            mock_doc = MagicMock()
            mock_doc.embed_documents.side_effect = fake_embed
            mock_query = MagicMock()
            mock_make.side_effect = [mock_doc, mock_query]

            from app.ai.rag.vector_store import GeminiDualTaskEmbeddings
            dual = GeminiDualTaskEmbeddings()
            original_batch_size = GeminiDualTaskEmbeddings.BATCH_SIZE
            GeminiDualTaskEmbeddings.BATCH_SIZE = 3

        texts = ["a"] * 7
        result = dual.embed_documents(texts)
        assert len(result) == 7
        assert mock_doc.embed_documents.call_count == 3  # ceil(7/3)

        GeminiDualTaskEmbeddings.BATCH_SIZE = original_batch_size


# ─── 3. sqlite-vec storage with Gemini-sized embeddings ─────────────────────

class TestSQLiteVecGeminiDimensions:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_768_dim_embeddings(self, tmp_path):
        """Verify sqlite-vec stores and retrieves 768-dim Gemini-sized embeddings."""
        from app.ai.rag.vector_store import SQLiteVectorStore

        db_file = tmp_path / "gemini_vec.db"
        emb = DeterministicFakeEmbedding(size=768)
        store = SQLiteVectorStore(db_path=db_file, embeddings_instance=emb)

        texts = [
            "AI market size is projected at $200B by 2028",
            "Key competitors include OpenAI, Anthropic, and Google",
        ]
        metadatas = [
            {"thread_id": "thread-gemini-1", "document_id": "doc-1"},
            {"thread_id": "thread-gemini-1", "document_id": "doc-2"},
        ]
        ids = await store.aadd_texts(texts, metadatas=metadatas)
        assert len(ids) == 2

        results = await store.asimilarity_search(
            "AI market size", k=1, filter={"thread_id": "thread-gemini-1"}
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_error(self, tmp_path):
        """Switching from 1536-dim to 768-dim without re-indexing should raise RuntimeError."""
        from app.ai.rag.vector_store import SQLiteVectorStore

        db_file = tmp_path / "mismatch_vec.db"

        # First store: 1536-dim (simulating old provider)
        emb_1536 = DeterministicFakeEmbedding(size=1536)
        store_1536 = SQLiteVectorStore(db_path=db_file, embeddings_instance=emb_1536)
        await store_1536.aadd_texts(["some text"], metadatas=[{"thread_id": "t1"}])

        # Second store: 768-dim (simulating Gemini migration)
        emb_768 = DeterministicFakeEmbedding(size=768)
        store_768 = SQLiteVectorStore(db_path=db_file, embeddings_instance=emb_768)

        with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
            store_768._ensure_tables()


# ─── 4. LLM routing ──────────────────────────────────────────────────────────

class TestLLMRouting:
    def test_mock_returned_without_any_key(self, monkeypatch):
        """No API keys → MockMarketResearchChatModel returned, no exception."""
        monkeypatch.setattr("app.core.config.settings.gemini_api_key",
                            __import__("pydantic").SecretStr(""))
        monkeypatch.setattr("app.core.config.settings.api_key",
                            __import__("pydantic").SecretStr("test_key"))
        from app.ai.graph.agent import create_model, MockMarketResearchChatModel
        model = create_model("gemini-2.0-flash")
        assert isinstance(model, MockMarketResearchChatModel)

    def test_gemini_model_initialized_with_key(self, monkeypatch):
        """With GEMINI_API_KEY set, create_model should return ChatGoogleGenerativeAI."""
        monkeypatch.setattr("app.core.config.settings.gemini_api_key",
                            __import__("pydantic").SecretStr("AIza-real-looking-key"))

        mock_chat_model = MagicMock(name="ChatGoogleGenerativeAI_instance")

        with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_chat_model) as mock_cls:
            from app.ai.graph.agent import create_model
            model = create_model("gemini-2.0-flash")
            mock_cls.assert_called_once()
            assert model is mock_chat_model

    def test_mock_tool_call_generation(self):
        """MockMarketResearchChatModel should generate a tool call on first message."""
        from app.ai.graph.agent import MockMarketResearchChatModel
        from langchain_core.outputs import ChatResult

        model = MockMarketResearchChatModel()
        result = model._generate([HumanMessage(content="What is the EV market size?")])
        assert isinstance(result, ChatResult)
        gen = result.generations[0].message
        assert gen.tool_calls, "Expected a tool call on first message"

    def test_mock_synthesis_after_tool_message(self):
        """A single retrieved fact should be quoted directly."""
        from app.ai.graph.agent import MockMarketResearchChatModel

        model = MockMarketResearchChatModel()
        messages = [
            HumanMessage(content="EV market overview"),
            ToolMessage(content="EV market grows at 25% CAGR", tool_call_id="call_1"),
        ]
        result = model._generate(messages)
        content = result.generations[0].message.content
        assert content == "EV market grows at 25% CAGR"


# ─── 5. End-to-end ingestion + retrieval (mocked Gemini) ──────────────────────

class TestEndToEndWithMockedGemini:
    @pytest.mark.asyncio
    async def test_index_and_retrieve_with_mock_768_embeddings(self, tmp_path):
        """Full ingestion → storage → retrieval cycle using 768-dim fake embeddings."""
        from app.ai.rag.vector_store import SQLiteVectorStore
        from uuid import uuid4

        db_file = tmp_path / "e2e.db"
        emb = DeterministicFakeEmbedding(size=768)
        store = SQLiteVectorStore(db_path=db_file, embeddings_instance=emb)

        doc_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = str(uuid4())

        texts = [
            "The global cloud computing market reached $650 billion in 2026.",
            "AWS holds approximately 32% cloud market share, followed by Azure at 22%.",
            "Multi-cloud strategies are adopted by 87% of enterprise organizations.",
        ]
        metas = [
            {"thread_id": thread_id, "document_id": doc_id, "user_id": user_id,
             "file_name": "cloud_report.pdf", "id": str(uuid4())},
            {"thread_id": thread_id, "document_id": doc_id, "user_id": user_id,
             "file_name": "cloud_report.pdf", "id": str(uuid4())},
            {"thread_id": thread_id, "document_id": doc_id, "user_id": user_id,
             "file_name": "cloud_report.pdf", "id": str(uuid4())},
        ]

        ids = await store.aadd_texts(texts, metadatas=metas)
        assert len(ids) == 3

        results = await store.asimilarity_search(
            "cloud market share", k=2, filter={"thread_id": thread_id}
        )
        assert len(results) >= 1, "Expected at least one result for 'cloud market share'"

        # Verify metadata propagation
        for r in results:
            assert r.metadata.get("thread_id") == thread_id
            assert r.metadata.get("document_id") == doc_id

    @pytest.mark.asyncio
    async def test_delete_restores_empty_store(self, tmp_path):
        """After deleting all chunks, similarity search returns nothing."""
        from app.ai.rag.vector_store import SQLiteVectorStore

        db_file = tmp_path / "delete_test.db"
        emb = DeterministicFakeEmbedding(size=768)
        store = SQLiteVectorStore(db_path=db_file, embeddings_instance=emb)

        ids = await store.aadd_texts(["Market data point 1"], metadatas=[{"thread_id": "t99"}])
        await store.adelete(ids)
        results = await store.asimilarity_search("market", k=1, filter={"thread_id": "t99"})
        assert len(results) == 0


# ─── 6. Live smoke test (only runs when GEMINI_API_KEY is set) ──────────────

def _is_gemini_key_available() -> bool:
    """Check if a real Gemini API key is available (via env or .env file)."""
    # Try os.environ first
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key and env_key not in ("", "test_gemini_key"):
        return True
    # Fall back to settings (which reads .env file)
    try:
        from app.core.config import settings
        return settings.active_provider == "google_genai"
    except Exception:
        return False


GEMINI_KEY_AVAILABLE = _is_gemini_key_available()


@pytest.mark.skipif(not GEMINI_KEY_AVAILABLE, reason="GEMINI_API_KEY not configured; skipping live smoke test")
class TestGeminiLiveSmoke:
    def test_live_embedding_dimension(self):
        """Call Gemini embedding API and verify returned dimension matches EMBEDDING_DIMENSIONS."""
        _reset_embedding_singleton()
        from app.ai.rag.vector_store import get_embeddings
        from app.core.config import settings

        emb = get_embeddings()
        vec = emb.embed_query("Market research test query")
        assert isinstance(vec, list)
        assert len(vec) == settings.embedding_dimensions, (
            f"Expected {settings.embedding_dimensions} dimensions, got {len(vec)}"
        )
        _reset_embedding_singleton()

    def test_live_document_embedding_dimension(self):
        """Verify document embedding dimension matches query embedding dimension."""
        _reset_embedding_singleton()
        from app.ai.rag.vector_store import get_embeddings, GeminiDualTaskEmbeddings
        from app.core.config import settings

        emb = get_embeddings()
        assert isinstance(emb, GeminiDualTaskEmbeddings)

        doc_vecs = emb.embed_documents(["AI industry report 2026"])
        query_vec = emb.embed_query("AI industry")
        assert len(doc_vecs[0]) == settings.embedding_dimensions
        assert len(query_vec) == settings.embedding_dimensions
        _reset_embedding_singleton()

    def test_live_gemini_chat_model_initializes(self):
        """Verify create_model returns a real Gemini model (not Mock) when key is set.

        Uses the fallback chain, so even if the user-configured model (e.g.
        gemini-3.8-flash) is temporarily 503, a fallback model should succeed.
        Does NOT invoke the model (avoids transient API failures flaking the test).
        """
        from app.core.config import settings
        from app.ai.graph.agent import create_model

        model = create_model(settings.gemini_model)
        model_type = type(model).__name__
        assert "Mock" not in model_type, (
            f"Expected real Gemini model but got {model_type}. "
            "All Gemini model candidates may be unavailable."
        )
        # Verify the model has bind_tools and invoke methods (LangGraph-compatible)
        assert hasattr(model, "bind_tools"), "Model must support bind_tools for LangGraph"
        assert hasattr(model, "invoke"), "Model must support invoke"

    def test_live_gemini_chat_model_responds(self):
        """Verify a Gemini chat model produces a non-empty response.

        Tolerates transient 503/504 from overloaded models — marks as xfail
        rather than hard-fail in those cases.
        """
        from app.core.config import settings
        from app.ai.graph.agent import create_model

        model = create_model(settings.gemini_model)
        if "Mock" in type(model).__name__:
            pytest.skip("No real Gemini model available (all candidates returned 503)")

        try:
            response = model.invoke([HumanMessage(content="Say: hello")])
            assert isinstance(response, AIMessage)
            content = str(response.content)
            assert len(content) > 0, "Expected non-empty response from Gemini"
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("503", "unavailable", "504", "deadline")):
                pytest.skip(f"Gemini model transiently unavailable: {str(e)[:100]}")
            raise

