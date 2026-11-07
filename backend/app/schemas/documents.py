from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    id: UUID | None = None
    file_name: str = Field(max_length=255)
    thread_id: UUID


# Alias to preserve backward compatibility for legacy typo
DocumetCreate = DocumentCreate


class DocumentPublic(BaseModel):
    id: UUID
    file_name: str
    thread_id: UUID
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    message: str


class DocumentDeleteResponse(BaseModel):
    message: str
