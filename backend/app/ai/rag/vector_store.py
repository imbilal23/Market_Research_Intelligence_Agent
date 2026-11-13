import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

import sqlite_vec
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger
from starlette.concurrency import run_in_threadpool

from app.core.config import settings

_embeddings: Optional[Embeddings] = None

# Gemini task_type constants for retrieval
# https://ai.google.dev/gemini-api/docs/embeddings#task-types
_GEMINI_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
_GEMINI_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


def _make_gemini_embeddings(task_type: str) -> "GoogleGenerativeAIEmbeddings":  # type: ignore[name-defined]
    """Construct a Gemini embedding instance for the given task type."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from pydantic import SecretStr

    raw_key = settings.gemini_api_key_str
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=SecretStr(raw_key),
        task_type=task_type,
        output_dimensionality=settings.embedding_dimensions,
        client_args={"timeout": 120},
    )


class GeminiDualTaskEmbeddings(Embeddings):
    """
    Wraps Gemini GoogleGenerativeAIEmbeddings with separate instances for
    document indexing (RETRIEVAL_DOCUMENT) and query embedding (RETRIEVAL_QUERY),
    as recommended by Google. Also implements bounded exponential-backoff retries
    for rate-limit and transient errors.
    """

    MAX_RETRIES = 4
    BATCH_SIZE = 50  # Process at most 50 chunks per API call

    def __init__(self) -> None:
        self._doc_embedder = _make_gemini_embeddings(_GEMINI_TASK_TYPE_DOCUMENT)
        self._query_embedder = _make_gemini_embeddings(_GEMINI_TASK_TYPE_QUERY)

    def _embed_with_retry(self, texts: list[str], embedder: Embeddings) -> list[list[float]]:
        """Embed texts with exponential backoff for rate-limit and transient errors."""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return embedder.embed_documents(texts)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in ("resource_exhausted", "quota", "rate limit", "429"))
                is_transient = any(kw in err_str for kw in ("unavailable", "503", "timeout", "connection"))

                if attempt < self.MAX_RETRIES and (is_rate_limit or is_transient):
                    wait = 2 ** attempt * (5 if is_rate_limit else 2)
                    retry_hint = re.search(r"(?:retrydelay[\"']?\s*:\s*[\"']?|retry in\s+)(\d+(?:\.\d+)?)s", err_str)
                    if retry_hint:
                        wait = max(wait, float(retry_hint.group(1)) + 1)
                    if wait > 120:
                        raise
                    logger.warning(
                        f"Gemini embedding {'rate limited' if is_rate_limit else 'transient error'} "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES}). Retrying in {wait}s."
                    )
                    time.sleep(wait)
                elif "api_key" in err_str or "invalid_argument" in err_str:
                    raise RuntimeError(
                        f"Invalid GEMINI_API_KEY or model configuration: {e}. "
                        "Verify your GEMINI_API_KEY in the environment."
                    ) from e
                else:
                    raise

        raise RuntimeError(f"Gemini embedding failed after {self.MAX_RETRIES} retries.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks in batches using RETRIEVAL_DOCUMENT task type."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            logger.debug(f"Embedding batch {i // self.BATCH_SIZE + 1} ({len(batch)} chunks) as RETRIEVAL_DOCUMENT")
            all_embeddings.extend(self._embed_with_retry(batch, self._doc_embedder))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using RETRIEVAL_QUERY task type."""
        result = self._embed_with_retry([text], self._query_embedder)
        return result[0]


def get_embeddings() -> Embeddings:
    """
    Return the active embedding instance.
    Priority:
      1. Gemini (GEMINI_API_KEY set) — uses GeminiDualTaskEmbeddings with task_type routing.
      2. Mock (offline/test) — DeterministicFakeEmbedding sized to settings.embedding_dimensions.
    OpenAI embeddings are not used when Gemini is configured.
    """
    global _embeddings
    if _embeddings is None:
        provider = settings.active_provider

        if provider == "google_genai":
            logger.info(
                f"Initializing Gemini embeddings: model='{settings.embedding_model}', "
                f"dimensions={settings.embedding_dimensions}"
            )
            _embeddings = GeminiDualTaskEmbeddings()

        else:
            logger.info(
                f"No valid GEMINI_API_KEY detected (provider='{provider}'). "
                f"Using DeterministicFakeEmbedding(size={settings.embedding_dimensions}) for offline/test mode."
            )
            _embeddings = DeterministicFakeEmbedding(size=settings.embedding_dimensions)

    return _embeddings


def reset_embeddings() -> None:
    """Force re-initialization of the embedding instance (used during re-indexing)."""
    global _embeddings
    _embeddings = None




class SQLiteVectorStore(VectorStore):
    """
    Production vector store implementation using SQLite and sqlite-vec.
    """

    def __init__(self, db_path: str | Path, embeddings_instance: Embeddings):
        self.db_path = str(db_path)
        self._embeddings_instance = embeddings_instance
        self._dim: Optional[int] = None
        self._initialized = False

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings_instance

    def _get_dim(self) -> int:
        if self._dim is None:
            if isinstance(self._embeddings_instance, GeminiDualTaskEmbeddings):
                self._dim = settings.embedding_dimensions
            else:
                self._dim = len(self._embeddings_instance.embed_query("dimension_test"))
        return self._dim

    def _get_connection(self) -> sqlite3.Connection:
        db_path = Path(self.db_path)
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.critical(
                f"Directory creation failed for vector store at '{db_path.parent}'. "
                f"Check persistent volume mount permissions (e.g. Railway /data). Error: {e}"
            )
            raise RuntimeError(
                f"Vector database directory not writable: '{db_path.parent}'. "
                f"Ensure volume mount permissions allow writing. Error: {e}"
            ) from e

        try:
            conn = sqlite3.connect(str(db_path))
        except sqlite3.Error as e:
            logger.critical(
                f"Failed to open SQLite vector database at '{db_path}'. "
                f"Check file permissions and storage path. Error: {e}"
            )
            raise RuntimeError(f"Failed to open SQLite database at '{db_path}': {e}") from e

        try:
            conn.enable_load_extension(True)
        except (AttributeError, sqlite3.OperationalError) as e:
            conn.close()
            logger.critical(
                "SQLite extension loading is not supported by Python sqlite3 in this environment. "
                "Ensure Python on Linux was compiled with loadable extension support. "
                f"Details: {e}"
            )
            raise RuntimeError(
                f"Python sqlite3 extension loading disabled: {e}. "
                "Ensure SQLite shared extension support is enabled in your Linux runtime."
            ) from e

        try:
            sqlite_vec.load(conn)
        except Exception as e:
            conn.close()
            ext_path = None
            try:
                ext_path = sqlite_vec.loadable_path()
            except Exception:
                pass
            logger.critical(
                f"Failed to load 'sqlite-vec' extension into SQLite database connection. "
                f"Attempted extension path: '{ext_path}'. Error: {e}. "
                "Action required: Verify that sqlite-vec is installed and matches the host Linux architecture (x86_64/arm64)."
            )
            raise RuntimeError(
                f"Failed to load sqlite-vec extension from '{ext_path}': {e}. "
                "Verify sqlite-vec package installation and OS architecture compatibility."
            ) from e

        return conn

    def check_health(self) -> tuple[bool, str]:
        """Verify that SQLite connection and sqlite-vec extension are operational."""
        try:
            with self._get_connection() as conn:
                res = conn.execute("SELECT vec_version()").fetchone()
                version = res[0] if res else "unknown"
                return True, f"sqlite-vec operational (version: {version})"
        except Exception as e:
            return False, f"sqlite-vec failed: {str(e)}"

    def _get_stored_dimension(self, conn: sqlite3.Connection) -> Optional[int]:
        """Return the dimension of vectors already stored in vec_documents, or None if empty."""
        try:
            row = conn.execute("SELECT vec_length(embedding) FROM vec_documents LIMIT 1").fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def _ensure_tables(self) -> None:
        if self._initialized:
            return
        dim = self._get_dim()
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    document_id TEXT,
                    thread_id TEXT,
                    user_id TEXT
                );
            """
            )

            # Check for existing vector dimension before creating the virtual table
            stored_dim = self._get_stored_dimension(conn)
            if stored_dim is not None and stored_dim != dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: stored vectors have dimension {stored_dim}, "
                    f"but the active embedding model produces dimension {dim} "
                    f"(model='{settings.embedding_model}', embedding_dimensions={settings.embedding_dimensions}). "
                    "Run 'python scripts/reindex_gemini.py' to rebuild the vector index with the new model, "
                    "or set EMBEDDING_DIMENSIONS to match the stored dimension."
                )

            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[{dim}]
                );
            """
            )
        self._initialized = True


    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        self._ensure_tables()
        text_list = list(texts)
        if not text_list:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_list]
        if metadatas is None:
            metadatas = [{} for _ in text_list]

        embeddings_list = self._embeddings_instance.embed_documents(text_list)
        if len(embeddings_list) != len(text_list):
            raise RuntimeError("The embedding service returned an incomplete result. Please retry.")

        with self._get_connection() as conn:
            for text, meta, doc_id, emb in zip(text_list, metadatas, ids, embeddings_list):
                document_id = meta.get("document_id")
                thread_id = meta.get("thread_id")
                user_id = meta.get("user_id")

                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents (id, text, metadata, document_id, thread_id, user_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        doc_id,
                        text,
                        json.dumps(meta),
                        str(document_id) if document_id is not None else None,
                        str(thread_id) if thread_id is not None else None,
                        str(user_id) if user_id is not None else None,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vec_documents (id, embedding)
                    VALUES (?, ?)
                """,
                    (doc_id, json.dumps(emb)),
                )

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        return await run_in_threadpool(self.add_texts, texts, metadatas=metadatas, ids=ids, **kwargs)

    async def aadd_documents(self, documents: list[Document], ids: Optional[list[str]] = None, **kwargs: Any) -> list[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [doc.metadata.get("id", str(uuid.uuid4())) for doc in documents]
        return await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        self._ensure_tables()

        if not query:
            # Metadata-only search (used for chunk retrieval & deletion without vector distance calculation)
            sql = "SELECT id, text, metadata FROM documents WHERE 1=1"
            params: list[Any] = []
            if filter:
                if "thread_id" in filter and filter["thread_id"]:
                    sql += " AND thread_id = ?"
                    params.append(str(filter["thread_id"]))
                if "document_id" in filter and filter["document_id"]:
                    sql += " AND document_id = ?"
                    params.append(str(filter["document_id"]))
                if "user_id" in filter and filter["user_id"]:
                    sql += " AND user_id = ?"
                    params.append(str(filter["user_id"]))
            with self._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [Document(page_content=row[1], metadata=json.loads(row[2])) for row in rows]

        query_vec = self._embeddings_instance.embed_query(query)

        # When filtering by metadata (e.g. thread_id), sqlite-vec KNN retrieves k nearest neighbors globally first.
        # To ensure thread/user-specific documents are not shadowed by other threads, expand candidate pool
        # and enforce top-k via outer LIMIT.
        knn_candidates = max(k * 50, 500) if filter else k

        sql = """
            SELECT d.id, d.text, d.metadata, v.distance
            FROM vec_documents v
            JOIN documents d ON v.id = d.id
            WHERE v.embedding MATCH ? AND k = ?
        """
        params = [json.dumps(query_vec), knn_candidates]

        if filter:
            if "thread_id" in filter and filter["thread_id"]:
                sql += " AND d.thread_id = ?"
                params.append(str(filter["thread_id"]))
            if "document_id" in filter and filter["document_id"]:
                sql += " AND d.document_id = ?"
                params.append(str(filter["document_id"]))
            if "user_id" in filter and filter["user_id"]:
                sql += " AND d.user_id = ?"
                params.append(str(filter["user_id"]))

        sql += " ORDER BY v.distance ASC LIMIT ?"
        params.append(k)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [Document(page_content=row[1], metadata=json.loads(row[2])) for row in rows]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        return await run_in_threadpool(self.similarity_search, query, k=k, filter=filter, **kwargs)

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return True
        self._ensure_tables()
        placeholders = ",".join("?" for _ in ids)
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM vec_documents WHERE id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)
        return True

    async def adelete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        return await run_in_threadpool(self.delete, ids=ids, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> "SQLiteVectorStore":
        db_path = kwargs.get("db_path", settings.vector_db_path)
        store = cls(db_path=db_path, embeddings_instance=embedding)
        store.add_texts(texts, metadatas=metadatas)
        return store


_vector_store_instance: Optional[SQLiteVectorStore] = None


def get_vector_store() -> SQLiteVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        emb = get_embeddings()
        _vector_store_instance = SQLiteVectorStore(db_path=settings.vector_db_path, embeddings_instance=emb)
    return _vector_store_instance


class LazyVectorStoreProxy:
    def __getattr__(self, name: str) -> Any:
        vs = get_vector_store()
        return getattr(vs, name)


vector_store = LazyVectorStoreProxy()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

DOCUMENT_LOADER_MAPPING: dict[str, type[BaseLoader]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

allowd_extensions = list(DOCUMENT_LOADER_MAPPING.keys())


async def _load_and_split_documents(file_path: Path) -> list[Document]:
    file_extension = file_path.suffix.lower()
    if file_extension not in DOCUMENT_LOADER_MAPPING:
        raise ValueError(f"Unsupported file type: {file_extension}, Allowed types: {', '.join(allowd_extensions)}")
    loader = DOCUMENT_LOADER_MAPPING[file_extension](file_path)  # type: ignore
    documents = await loader.aload()
    splits = text_splitter.split_documents(documents)
    logger.info(f"Successfully loaded and split {file_path} into {len(splits)} chunks.")
    return splits


async def index_document_to_vectorstore(
    file_path: Path, document_id: UUID, thread_id: UUID, user_id: UUID, file_name: Optional[str] = None
) -> list[str]:
    display_name = file_name or file_path.name
    logger.info(f"Starting indexing for document: {file_path} (name: {display_name}) with document_id: {document_id}")
    splits = await _load_and_split_documents(file_path)
    if not splits:
        raise ValueError("The file contains no readable text.")
    for split in splits:
        split.metadata["id"] = str(uuid.uuid4())
        split.metadata["file_name"] = display_name
        split.metadata["document_id"] = str(document_id)
        split.metadata["thread_id"] = str(thread_id)
        split.metadata["user_id"] = str(user_id)

    try:
        vs = get_vector_store()
        doc_ids = await vs.aadd_documents(splits, ids=[split.metadata["id"] for split in splits])
        logger.info(
            f"Successfully indexed {len(splits)} chunks for document {file_path} (document_id: {document_id}) to SQLiteVectorStore."
        )
        return doc_ids
    except Exception as e:
        logger.error(f"Error adding documents to VectorStore: {e}")
        raise


async def search_documents_in_vectorstore(query: str = "", k: int = 1, filter: dict | None = None) -> list[Document]:
    logger.info(f"Search documents with query: '{query}', k: {k}, filter: {filter} in VectorStore.")
    vs = get_vector_store()
    documents = await vs.asimilarity_search(query, k=k, filter=filter)
    if not documents:
        logger.info(f"No documents found for query: '{query}' and filter: {filter} in VectorStore.")
    else:
        logger.info(f"Found {len(documents)} document chunks for query: '{query}' and filter: {filter} in VectorStore.")
    return documents


async def delete_document_from_vectorstore(document_ids: list[str]) -> None:
    logger.info(f"Attempting to delete {len(document_ids)} document chunks from VectorStore.")
    vs = get_vector_store()
    await vs.adelete(ids=document_ids)
    logger.info(f"Successfully deleted {len(document_ids)} document chunks from VectorStore.")


# Backward compatibility aliases
index_document_to_pgvector = index_document_to_vectorstore
search_documents_in_pgvector = search_documents_in_vectorstore
delete_document_from_pgvector = delete_document_from_vectorstore
