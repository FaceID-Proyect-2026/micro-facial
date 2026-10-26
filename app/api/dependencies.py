from secrets import compare_digest

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if (not settings.api_key.strip() or not api_key
            or not compare_digest(api_key.encode("utf-8"), settings.api_key.encode("utf-8"))):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key invalida o ausente.",
        )


async def db_session() -> AsyncSession:
    async for session in get_session():
        yield session


DatabaseSession = Depends(db_session)
ApiKeyGuard = Depends(require_api_key)
