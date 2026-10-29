from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.api.dependencies import ApiKeyGuard, facial_service
from app.schemas.facial_embedding import (
    FacialEmbeddingCreate,
    FacialEmbeddingFromImageCreate,
    FacialEmbeddingPartialUpdate,
    FacialEmbeddingResponse,
    FacialEmbeddingSummary,
    FacialLivenessCheckRequest,
    FacialLivenessCheckResponse,
    FacialVerificationRequest,
    FacialVerificationResponse,
)
from app.services.facial_embedding_service import DuplicateFaceError, FacialEmbeddingService

router = APIRouter(dependencies=[ApiKeyGuard])


@router.post(
    "",
    response_model=FacialEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Guardar embedding facial del aprendiz",
)
async def register_facial_embedding(
    payload: FacialEmbeddingCreate,
    service: FacialEmbeddingService = Depends(facial_service),
) -> FacialEmbeddingResponse:
    try:
        return await service.register(payload)
    except DuplicateFaceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/from-image",
    response_model=FacialEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generar y guardar embedding facial desde una imagen",
)
async def register_facial_embedding_from_image(
    payload: FacialEmbeddingFromImageCreate,
    service: FacialEmbeddingService = Depends(facial_service),
) -> FacialEmbeddingResponse:
    try:
        return await service.register_from_image(payload)
    except DuplicateFaceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/verify-session",
    response_model=FacialVerificationResponse,
    summary="Verificar rostro temporal contra los aprendices de una sesion",
)
async def verify_session_face(
    payload: FacialVerificationRequest,
    service: FacialEmbeddingService = Depends(facial_service),
) -> FacialVerificationResponse:
    try:
        return await service.verify_session_face(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/liveness-check",
    response_model=FacialLivenessCheckResponse,
    summary="Validar un reto de vida puntual sin registrar ni comparar identidad",
)
async def check_liveness(
    payload: FacialLivenessCheckRequest,
    service: FacialEmbeddingService = Depends(facial_service),
) -> FacialLivenessCheckResponse:
    try:
        return service.check_liveness(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/users/{user_id}",
    response_model=FacialEmbeddingResponse,
    summary="Consultar embedding facial activo por aprendiz",
)
async def get_active_facial_embedding(
    user_id: UUID,
    service: FacialEmbeddingService = Depends(facial_service),
) -> FacialEmbeddingResponse:
    record = await service.get_active_by_user(user_id)
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
    service: FacialEmbeddingService = Depends(facial_service),
) -> FacialEmbeddingResponse:
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
    service: FacialEmbeddingService = Depends(facial_service),
) -> list[FacialEmbeddingSummary]:
    return await service.list_active(limit=limit, offset=offset)


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Desactivar embedding facial activo",
)
async def deactivate_facial_embedding(
    user_id: UUID,
    deleted_by: str = Query(default="embedding-service", max_length=100),
    service: FacialEmbeddingService = Depends(facial_service),
) -> Response:
    deleted = await service.deactivate(user_id, deleted_by)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El aprendiz no tiene embedding facial activo.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
