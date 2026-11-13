import shutil
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from loguru import logger
from starlette.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.rag.vector_store import (
    DOCUMENT_LOADER_MAPPING,
    delete_document_from_vectorstore,
    index_document_to_vectorstore,
    search_documents_in_vectorstore,
)
from app.core.config import settings
from app.core.provider_errors import provider_error_message
from app.db.models import Document
from app.schemas.documents import DocumentCreate
from app.services.thread_service import get_thread


async def get_documents(thread_id: UUID, user_id: UUID, session: AsyncSession) -> Sequence[Document]:
    await get_thread(thread_id, user_id, session)
    statement = select(Document).where(Document.thread_id == thread_id)
    result = await session.execute(statement)
    return result.scalars().all()


async def insert_document(document_data: DocumentCreate, session: AsyncSession) -> Document:
    new_document = Document(**document_data.model_dump(exclude_none=True))
    session.add(new_document)
    await session.commit()
    return new_document


async def upload_and_index_document(
    thread_id: UUID, file: UploadFile, user_id: UUID, session: AsyncSession
) -> dict:
    await get_thread(thread_id, user_id, session)
    if file.filename is None:
        logger.error("No file uploaded.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    if "/" in file.filename or "\\" in file.filename or ":" in file.filename or "\x00" in file.filename:
        raise HTTPException(status_code=400, detail="Upload a filename without path components.")
    if file.size == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if file.size is not None and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Files must be 50 MiB or smaller.")

    allowed_extensions = list(DOCUMENT_LOADER_MAPPING.keys())
    message = f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
    if Path(file.filename).suffix.lower() not in allowed_extensions:
        logger.error(message)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    # Ensure persistent uploads directory exists (e.g. /data/uploads or local)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = uuid4()
    persistent_filename = f"{document_id}_{file.filename}"
    persistent_file_path = upload_dir / persistent_filename

    chunk_ids = []
    try:
        with open(persistent_file_path, "wb") as buffer:
            await run_in_threadpool(shutil.copyfileobj, file.file, buffer)
        logger.info(f"File '{file.filename}' stored persistently at '{persistent_file_path}'.")

        chunk_ids = await index_document_to_vectorstore(
            persistent_file_path,
            document_id,
            thread_id,
            user_id,
            file_name=file.filename,
        )
        await get_thread(thread_id, user_id, session)
        document_data = DocumentCreate(id=document_id, file_name=file.filename, thread_id=thread_id)
        await insert_document(document_data, session)
        logger.info(f"File '{file.filename}' (document_id: {document_id}) indexed to VectorStore ({len(chunk_ids)} chunks).")

        return {"document_id": document_id, "message": f"File {file.filename} uploaded, persisted, and indexed successfully."}
    except Exception as e:
        logger.error("Document upload failed ({})", type(e).__name__)
        await session.rollback()
        # Cleanup persistent file on error
        if persistent_file_path.exists():
            try:
                persistent_file_path.unlink()
            except Exception:
                pass
        if chunk_ids:
            try:
                await delete_document_from_vectorstore(chunk_ids)
                logger.info(f"Cleaned up VectorStore chunks for document {document_id} after upload failure.")
            except Exception as clean_err:
                logger.error(f"Failed to cleanup VectorStore for document {document_id}: {clean_err}")
        try:
            db_doc = await session.get(Document, document_id)
            if db_doc:
                await session.delete(db_doc)
                await session.commit()
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=provider_error_message(e, "We could not process this file. Check that it contains readable text and try again."),
        )


async def delete_document_with_chunks(document_id: UUID, user_id: UUID, session: AsyncSession) -> dict:
    """Delete a document from database, VectorStore, and persistent volume storage."""
    db_document = await session.get(Document, document_id)
    if db_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found.",
        )

    await get_thread(db_document.thread_id, user_id, session)

    # 1. Clean up VectorStore chunks
    chunk_message = ""
    document_chunks = await search_documents_in_vectorstore(filter={"document_id": str(document_id)})
    if not document_chunks:
        logger.warning(f"Document chunks related to document {document_id} not found in VectorStore.")
    else:
        doc_ids_to_delete = [doc.metadata["id"] for doc in document_chunks]
        await delete_document_from_vectorstore(doc_ids_to_delete)
        logger.info(f"Successfully deleted all document chunks related to document_id: {document_id}.")
        chunk_message = "all its chunks from VectorStore"

    # 2. Clean up persistent file from upload_dir
    upload_dir = Path(settings.upload_dir)
    matched_files = list(upload_dir.glob(f"{document_id}_*"))
    for f in matched_files:
        try:
            f.unlink()
            logger.info(f"Removed persistent file '{f.name}' from storage.")
        except Exception as file_err:
            logger.warning(f"Failed to remove persistent file '{f.name}': {file_err}")

    # 3. Clean up database record
    await session.delete(db_document)
    await session.commit()
    logger.info(f"Successfully deleted document {document_id} from database.")

    return {"message": f"Successfully deleted document {document_id} from database and {chunk_message}."}
