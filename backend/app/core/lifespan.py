from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.ai.rag.vector_store import vector_store
from app.db.checkpointer import close_connection, create_connection
from app.db.session import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running lifespan before application startup...")
    try:
        await init_db()
        await create_connection()
        vector_store._ensure_tables()
        logger.info("SQLite application, checkpoint and vector storage initialized")
        yield
    finally:
        await close_connection()
        await close_db()
