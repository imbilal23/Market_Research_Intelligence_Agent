from collections.abc import AsyncGenerator
from pathlib import Path

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.models import Base

engine: AsyncEngine = create_async_engine(url=settings.database_uri)
async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def init_db() -> None:
    db_path = Path(settings.sqlite_app_db_path)
    logger.info(f"Initializing SQLite application database at '{db_path}'...")
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as dir_err:
        logger.critical(
            f"Failed to create directory for database at '{db_path.parent}'. "
            f"Verify volume mount path and permissions. Details: {dir_err}"
        )
        raise RuntimeError(
            f"Database directory '{db_path.parent}' not accessible: {dir_err}. "
            "Please check SQLITE_APP_DB_PATH and storage volume configuration."
        ) from dir_err

    try:
        async with engine.begin() as conn:
            # CREATE TABLE IF NOT EXISTS preserves existing data across container restarts
            await conn.run_sync(Base.metadata.create_all)
            logger.info(f"✅ Application database initialized successfully at '{db_path}' (existing data preserved)")
    except Exception as e:
        logger.critical(
            f"Database table creation failed for '{db_path}'. "
            f"Check file write permissions on the mounted storage volume. Error: {e}"
        )
        raise RuntimeError(
            f"Database initialization failed at '{db_path}': {e}. "
            "Verify SQLITE_APP_DB_PATH, volume write permissions, and disk space."
        ) from e


async def check_db_health() -> tuple[bool, str]:
    """Lightweight connectivity probe for the health check endpoint."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "Database connected"
    except Exception as e:
        return False, f"Database connectivity error: {str(e)}"


async def close_db() -> None:
    try:
        await engine.dispose()
        logger.info("✅ Database engine disposed successfully")
    except Exception as e:
        logger.warning(f"Error disposing database engine: {e}")
