from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from loguru import logger

from app.core.config import settings

checkpointer: AsyncSqliteSaver | None = None
_saver_cm = None


async def create_connection() -> AsyncSqliteSaver:
    global checkpointer, _saver_cm

    if checkpointer is not None:
        return checkpointer

    path = Path(settings.sqlite_checkpointer_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as dir_err:
        logger.critical(
            f"Failed to create directory for LangGraph checkpointer at '{path.parent}'. "
            f"Error: {dir_err}"
        )
        raise RuntimeError(
            f"Checkpointer directory '{path.parent}' not accessible: {dir_err}. "
            "Please verify SQLITE_CHECKPOINTER_PATH and volume mount permissions."
        ) from dir_err

    logger.info(f"Opening LangGraph SQLite checkpointer connection at '{path}'...")
    try:
        _saver_cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_uri)
        checkpointer = await _saver_cm.__aenter__()
        await checkpointer.setup()
        logger.info(f"✅ SQLite Checkpointer initialized successfully at '{path}' (history preserved)")
        return checkpointer
    except Exception as e:
        logger.critical(
            f"Failed to initialize SQLite checkpointer at '{path}'. "
            f"Verify volume write permissions and file health. Error: {e}"
        )
        raise RuntimeError(
            f"Checkpointer initialization failed at '{path}': {e}. "
            "Ensure the storage path is writable."
        ) from e


async def get_checkpointer() -> AsyncSqliteSaver:
    return await create_connection()


async def close_connection() -> None:
    global checkpointer, _saver_cm

    if _saver_cm is not None:
        logger.info("Closing SQLite checkpointer connection...")
        try:
            await _saver_cm.__aexit__(None, None, None)
            logger.info("✅ SQLite Checkpointer connection closed successfully")
        except Exception as e:
            logger.warning(f"Error closing SQLite checkpointer connection: {e}")
        finally:
            checkpointer = None
            _saver_cm = None
