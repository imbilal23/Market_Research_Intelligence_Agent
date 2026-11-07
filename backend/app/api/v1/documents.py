from uuid import UUID

from fastapi import APIRouter, UploadFile

from app.core.deps import CurrentUserDep, SessionDep
from app.schemas.documents import DocumentDeleteResponse, DocumentPublic, DocumentUploadResponse
from app.services import document_service

document_router = APIRouter()


@document_router.get("/{thread_id}", response_model=list[DocumentPublic])
async def get_documents(thread_id: UUID, current_user: CurrentUserDep, session: SessionDep):
    return await document_service.get_documents(thread_id, current_user.id, session)


@document_router.post("/upload/{thread_id}", response_model=DocumentUploadResponse)
async def upload_document(thread_id: UUID, file: UploadFile, current_user: CurrentUserDep, session: SessionDep):
    return await document_service.upload_and_index_document(thread_id, file, current_user.id, session)


@document_router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: UUID, current_user: CurrentUserDep, session: SessionDep):
    return await document_service.delete_document_with_chunks(document_id, current_user.id, session)
