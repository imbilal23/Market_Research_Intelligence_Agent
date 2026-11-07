import pytest
from app.core.security import hash_password, verify_password, create_jwt_token, decode_token
from app.ai.rag.vector_store import SQLiteVectorStore
from langchain_community.embeddings import DeterministicFakeEmbedding


def test_password_hashing():
    pwd = "testpassword123"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrongpassword", hashed) is False


def test_jwt_token_encode_decode():
    user_data = {"email": "test@example.com", "id": "00000000-0000-0000-0000-000000000001"}
    token = create_jwt_token(user_data, refresh=False)
    decoded = decode_token(token)
    assert decoded is not None
    assert decoded.user.email == "test@example.com"
    assert decoded.refresh is False


@pytest.mark.asyncio
async def test_sqlite_vector_store(tmp_path):
    db_file = tmp_path / "test_vec.db"
    emb = DeterministicFakeEmbedding(size=64)
    store = SQLiteVectorStore(db_path=db_file, embeddings_instance=emb)

    # Ingestion
    texts = ["Global market intelligence report 2026", "Quarterly financial forecast"]
    metadatas = [{"thread_id": "thread-1"}, {"thread_id": "thread-2"}]
    ids = await store.aadd_texts(texts, metadatas=metadatas)
    assert len(ids) == 2

    # Similarity search
    results = await store.asimilarity_search("market intelligence", k=1, filter={"thread_id": "thread-1"})
    assert len(results) == 1
    assert "Global market intelligence" in results[0].page_content

    # Filtered search non-matching
    no_results = await store.asimilarity_search("market intelligence", k=1, filter={"thread_id": "thread-999"})
    assert len(no_results) == 0

    # Deletion
    deleted = await store.adelete(ids)
    assert deleted is True

    # After deletion
    after_results = await store.asimilarity_search("market intelligence", k=1, filter={"thread_id": "thread-1"})
    assert len(after_results) == 0
