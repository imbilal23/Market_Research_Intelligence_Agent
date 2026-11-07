"""
scripts/reindex_gemini.py
=========================
Safe vector re-indexing script for the Market Research Intelligence Agent.

Use this script after switching to a new embedding provider or when
EMBEDDING_DIMENSIONS changes (e.g. migrating from 1536-dim fake/OpenAI
embeddings to 768-dim Gemini embeddings).

What it does:
  1. Reads all existing document chunks from the 'documents' SQLite table
     (text + metadata), preserving originals.
  2. Drops and recreates the vec_documents virtual table with the new dimension.
  3. Re-embeds all chunks using the active embedding model (Gemini if configured).
  4. Reinserts the new embeddings into the fresh vec_documents table.

What it does NOT do:
  - Touch the application database (app.db) or its metadata records.
  - Touch the LangGraph checkpointer (checkpoints.db) or conversation history.
  - Re-read or re-parse the original uploaded files.

Requirements:
  - GEMINI_API_KEY must be set in the environment or .env file.
  - Run from the project root:
      python scripts/reindex_gemini.py
"""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

# Add backend to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from loguru import logger


async def reindex() -> None:
    from app.core.config import settings
    from app.ai.rag.vector_store import get_embeddings, reset_embeddings, GeminiDualTaskEmbeddings

    vector_db = Path(settings.vector_db_path)
    if not vector_db.exists():
        logger.error(f"Vector database not found at '{vector_db}'. Nothing to re-index.")
        return

    # ---- Step 1: Check active provider ----
    provider = settings.active_provider
    logger.info(f"Active provider: '{provider}'")
    logger.info(f"Embedding model: '{settings.embedding_model}'")
    logger.info(f"Target dimensions: {settings.embedding_dimensions}")

    if provider != "google_genai":
        logger.error(
            "GEMINI_API_KEY is not set or invalid. Cannot re-index with Gemini embeddings. "
            "Set GEMINI_API_KEY in your .env file and retry."
        )
        sys.exit(1)

    # Force fresh embedding instance after changing config
    reset_embeddings()
    embeddings = get_embeddings()

    if not isinstance(embeddings, GeminiDualTaskEmbeddings):
        logger.error("Expected GeminiDualTaskEmbeddings but got a different type. Check GEMINI_API_KEY.")
        sys.exit(1)

    # ---- Step 2: Read existing document chunks ----
    logger.info(f"Reading document chunks from '{vector_db}'...")

    import sqlite_vec
    conn = sqlite3.connect(str(vector_db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Check stored dimension
    stored_dim_row = conn.execute(
        "SELECT vec_length(embedding) FROM vec_documents LIMIT 1"
    ).fetchone()
    stored_dim = stored_dim_row[0] if stored_dim_row else None
    target_dim = settings.embedding_dimensions

    if stored_dim == target_dim:
        logger.info(
            f"Stored dimension ({stored_dim}) already matches target ({target_dim}). "
            "Re-indexing anyway to update embeddings to the new model."
        )

    chunks = conn.execute("SELECT id, text, metadata, document_id, thread_id, user_id FROM documents").fetchall()
    logger.info(f"Found {len(chunks)} document chunks to re-index.")

    if not chunks:
        logger.warning("No document chunks found. Nothing to do.")
        conn.close()
        return

    # ---- Step 3: Drop and recreate vec_documents with new dimension ----
    logger.info(f"Recreating vec_documents table with dimension={target_dim}...")
    conn.execute("DROP TABLE IF EXISTS vec_documents")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE vec_documents USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{target_dim}]
        );
        """
    )
    conn.commit()
    logger.info("✅ vec_documents table recreated.")

    # ---- Step 4: Re-embed and insert in batches ----
    BATCH_SIZE = 20
    total = len(chunks)
    ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]

    logger.info(f"Re-embedding {total} chunks in batches of {BATCH_SIZE}...")
    all_embeddings = []

    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i: i + BATCH_SIZE]
        batch_texts = texts[i: i + BATCH_SIZE]
        logger.info(f"  Batch {i // BATCH_SIZE + 1}: chunks {i + 1}-{min(i + BATCH_SIZE, total)} / {total}")
        batch_embeddings = embeddings.embed_documents(batch_texts)

        if len(batch_embeddings[0]) != target_dim:
            logger.error(
                f"Embedding dimension mismatch: expected {target_dim}, got {len(batch_embeddings[0])}. "
                "Check EMBEDDING_DIMENSIONS."
            )
            conn.close()
            sys.exit(1)

        for chunk_id, emb in zip(batch_ids, batch_embeddings):
            conn.execute(
                "INSERT INTO vec_documents (id, embedding) VALUES (?, ?)",
                [chunk_id, json.dumps(emb)],
            )
        conn.commit()
        all_embeddings.extend(batch_embeddings)
        logger.info(f"  ✅ Inserted {len(batch_ids)} embeddings.")

    conn.close()

    # ---- Step 5: Verify ----
    conn2 = sqlite3.connect(str(vector_db))
    conn2.enable_load_extension(True)
    sqlite_vec.load(conn2)
    final_count = conn2.execute("SELECT count(*) FROM vec_documents").fetchone()[0]
    final_dim = conn2.execute("SELECT vec_length(embedding) FROM vec_documents LIMIT 1").fetchone()
    final_dim_val = final_dim[0] if final_dim else "?"
    conn2.close()

    logger.info("=" * 60)
    logger.info(f"🎉 Re-indexing complete!")
    logger.info(f"   Chunks re-embedded : {final_count}")
    logger.info(f"   New dimension      : {final_dim_val}")
    logger.info(f"   Embedding model    : {settings.embedding_model}")
    logger.info(f"   Vector database    : {vector_db}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(reindex())
