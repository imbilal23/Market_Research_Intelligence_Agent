from pathlib import Path

from fastapi import APIRouter, Response, status

from app.ai.rag.vector_store import vector_store
from app.core.config import settings
from app.db import checkpointer as checkpointer_module
from app.db.session import check_db_health

health_router = APIRouter()


@health_router.get("", status_code=status.HTTP_200_OK)
@health_router.get("/", status_code=status.HTTP_200_OK)
async def check_system_health(response: Response):
    """
    Lightweight health check endpoint for hosting platforms (Railway, Docker, load balancers).
    Verifies SQLite application database, LangGraph checkpointer, and sqlite-vec vector store.
    """
    db_ok, db_msg = await check_db_health()
    vec_ok, vec_msg = vector_store.check_health()
    cp_ready = checkpointer_module.checkpointer is not None

    app_db_path = Path(settings.sqlite_app_db_path)
    cp_path = Path(settings.sqlite_checkpointer_path)
    vec_path = Path(settings.sqlite_vector_db_path)
    upload_path = Path(settings.upload_dir)

    all_healthy = db_ok and vec_ok and cp_ready

    payload = {
        "status": "healthy" if all_healthy else "degraded",
        "service": "market-research-intelligence-agent",
        "database": {
            "status": "connected" if db_ok else "error",
            "message": db_msg,
            "path": str(app_db_path),
            "exists": app_db_path.exists(),
        },
        "vector_search": {
            "status": "ready" if vec_ok else "error",
            "message": vec_msg,
            "path": str(vec_path),
            "exists": vec_path.exists(),
        },
        "checkpointer": {
            "status": "connected" if cp_ready else "not_connected",
            "path": str(cp_path),
            "exists": cp_path.exists(),
        },
        "storage": {
            "data_dir": str(settings.data_dir),
            "upload_dir": str(upload_path),
            "upload_dir_exists": upload_path.exists(),
        },
    }

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return payload
