from fastapi import APIRouter

from app.api.v1.auth import auth_router
from app.api.v1.chat import chat_router
from app.api.v1.documents import document_router
from app.api.v1.health import health_router
from app.api.v1.threads import thread_router
from app.api.v1.users import user_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, prefix="/health", tags=["HEALTH"])
api_v1_router.include_router(auth_router, prefix="/auth", tags=["AUTH"])
api_v1_router.include_router(user_router, prefix="/users", tags=["USERS"])
api_v1_router.include_router(thread_router, prefix="/threads", tags=["THREADS"])
api_v1_router.include_router(chat_router, prefix="/chat", tags=["CHAT"])
api_v1_router.include_router(document_router, prefix="/documents", tags=["DOCUMENTS"])
