from secrets import compare_digest
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.services.facial_embedding_service import FacialEmbeddingService
from app.repositories.facial_embedding_repository import FacialEmbeddingRepository
from app.services.liveness_validator import LivenessValidator
from app.workers.face_embedding_worker import get_face_embedding_worker

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    expected = settings.api_key.strip()
    if not expected or expected == "change-me" or not api_key or not compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key invalida o ausente.",
        )


async def db_session() -> AsyncSession:
    async for session in get_session():
        yield session


def facial_service(session: AsyncSession = Depends(db_session)) -> FacialEmbeddingService:
    """Compose request-scoped application dependencies outside the HTTP handlers."""
    return FacialEmbeddingService(
        session,
        embedding_worker=get_face_embedding_worker(),
        repository=FacialEmbeddingRepository(session),
        liveness_validator=LivenessValidator(),
    )


DatabaseSession = Depends(db_session)
ApiKeyGuard = Depends(require_api_key)
