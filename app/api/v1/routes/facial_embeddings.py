from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import ApiKeyGuard, db_session
from app.schemas.facial_embedding import (
    FacialEmbeddingCreate,
    FacialEmbeddingFromImageCreate,
    FacialEmbeddingPartialUpdate,
    FacialEmbeddingResponse,
    FacialEmbeddingSummary,
)
from app.services.facial_embedding_service import FacialEmbeddingService

router = APIRouter(dependencies=[ApiKeyGuard])


@router.post(
    "",
    response_model=FacialEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Guardar embedding facial del aprendiz",
)
async def register_facial_embedding(
    payload: FacialEmbeddingCreate,
    session: AsyncSession = Depends(db_session),
) -> FacialEmbeddingResponse:
    service = FacialEmbeddingService(session)
    try:
        return await service.register(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post(
    "/from-image",
    response_model=FacialEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generar y guardar embedding facial desde una imagen",
)
async def register_facial_embedding_from_image(
    payload: FacialEmbeddingFromImageCreate,
    session: AsyncSession = Depends(db_session),
) -> FacialEmbeddingResponse:
    service = FacialEmbeddingService(session)
    try:
        return await service.register_from_image(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get(
    "/users/{user_id}",
    response_model=FacialEmbeddingResponse,
    summary="Consultar embedding facial activo por aprendiz",
)
async def get_active_facial_embedding(
    user_id: UUID,
    session: AsyncSession = Depends(db_session),
) -> FacialEmbeddingResponse:
    record = await FacialEmbeddingService(session).get_active_by_user(user_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El aprendiz no tiene embedding facial activo.",
        )
    return record


@router.patch(
    "/users/{user_id}",
    response_model=FacialEmbeddingResponse,
    summary="Actualizar parcialmente embedding facial activo por aprendiz",
)
async def update_facial_embedding_partial(
    user_id: UUID,
    payload: FacialEmbeddingPartialUpdate,
    session: AsyncSession = Depends(db_session),
) -> FacialEmbeddingResponse:
    service = FacialEmbeddingService(session)
    try:
        return await service.update_partial(user_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "",
    response_model=list[FacialEmbeddingSummary],
    summary="Listar embeddings faciales activos",
)
async def list_active_facial_embeddings(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(db_session),
) -> list[FacialEmbeddingSummary]:
    return await FacialEmbeddingService(session).list_active(limit=limit, offset=offset)


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Desactivar embedding facial activo",
)
async def deactivate_facial_embedding(
    user_id: UUID,
    deleted_by: str = Query(default="embedding-service", max_length=100),
    session: AsyncSession = Depends(db_session),
) -> Response:
    deleted = await FacialEmbeddingService(session).deactivate(user_id, deleted_by)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El aprendiz no tiene embedding facial activo.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
