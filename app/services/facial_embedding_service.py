from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.facial_embedding_repository import FacialEmbeddingRepository
from app.schemas.facial_embedding import (
    FacialEmbeddingCreate,
    FacialEmbeddingFromImageCreate,
    FacialEmbeddingPartialUpdate,
    FacialEmbeddingResponse,
    FacialEmbeddingSummary,
)
from app.workers.face_embedding_worker import FaceEmbeddingWorker, get_face_embedding_worker


class FacialEmbeddingService:
    def __init__(
        self,
        session: AsyncSession,
        embedding_worker: FaceEmbeddingWorker | None = None,
    ) -> None:
        self.session = session
        self.repository = FacialEmbeddingRepository(session)
        self.embedding_worker = embedding_worker or get_face_embedding_worker()

    async def register(self, payload: FacialEmbeddingCreate) -> FacialEmbeddingResponse:
        return await self._register_embedding(payload)

    async def register_from_image(self, payload: FacialEmbeddingFromImageCreate) -> FacialEmbeddingResponse:
        generated = self.embedding_worker.generate(payload.image_base64)
        embedding_payload = FacialEmbeddingCreate(
            user_id=payload.user_id,
            embedding=generated.embedding,
            model_name=generated.model_name,
            photo_reference=payload.photo_reference,
            replace_existing=payload.replace_existing,
            created_by=payload.created_by,
        )
        return await self._register_embedding(embedding_payload)

    async def _register_embedding(self, payload: FacialEmbeddingCreate) -> FacialEmbeddingResponse:
        if not await self.repository.user_exists(payload.user_id):
            raise LookupError("No existe un aprendiz/usuario con ese user_id.")

        active = await self.repository.get_active_by_user(payload.user_id)
        if active and not payload.replace_existing:
            raise FileExistsError("El aprendiz ya tiene un embedding facial activo.")

        try:
            if active:
                await self.repository.replace_active(payload.user_id, payload.created_by)

            row = await self.repository.create(
                user_id=payload.user_id,
                embedding=payload.embedding,
                model_name=payload.model_name,
                photo_reference=payload.photo_reference,
                created_by=payload.created_by,
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

        return FacialEmbeddingResponse.model_validate(row)

    async def get_active_by_user(self, user_id: UUID) -> FacialEmbeddingResponse | None:
        row = await self.repository.get_active_by_user(user_id)
        return FacialEmbeddingResponse.model_validate(row) if row else None

    async def update_partial(
        self,
        user_id: UUID,
        payload: FacialEmbeddingPartialUpdate,
    ) -> FacialEmbeddingResponse:
        editable_fields = {"embedding", "image_base64", "model_name", "photo_reference"}
        if not payload.model_fields_set.intersection(editable_fields):
            raise ValueError("Debe enviar al menos un campo para actualizar.")
        if "embedding" in payload.model_fields_set and payload.embedding is None:
            raise ValueError("El embedding no puede ser nulo.")
        if "image_base64" in payload.model_fields_set and payload.image_base64 is None:
            raise ValueError("La imagen no puede ser nula.")
        if payload.embedding is not None and payload.image_base64 is not None:
            raise ValueError("Envia embedding o image_base64, no ambos.")
        if "model_name" in payload.model_fields_set and payload.model_name is None:
            raise ValueError("model_name no puede ser nulo.")

        if not await self.repository.user_exists(user_id):
            raise LookupError("No existe un aprendiz/usuario con ese user_id.")

        embedding = payload.embedding if "embedding" in payload.model_fields_set else None
        model_name = payload.model_name if "model_name" in payload.model_fields_set else None
        if payload.image_base64 is not None:
            generated = self.embedding_worker.generate(payload.image_base64)
            embedding = generated.embedding
            model_name = model_name or generated.model_name

        try:
            row = await self.repository.update_active(
                user_id=user_id,
                embedding=embedding,
                model_name=model_name,
                photo_reference=payload.photo_reference,
                update_photo_reference="photo_reference" in payload.model_fields_set,
                updated_by=payload.updated_by,
            )
            if row is None:
                raise LookupError("El aprendiz no tiene embedding facial activo.")
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

        return FacialEmbeddingResponse.model_validate(row)

    async def list_active(self, limit: int, offset: int) -> list[FacialEmbeddingSummary]:
        rows = await self.repository.list_active(limit=limit, offset=offset)
        return [FacialEmbeddingSummary.model_validate(row) for row in rows]

    async def deactivate(self, user_id: UUID, deleted_by: str) -> bool:
        try:
            deleted = await self.repository.deactivate(user_id, deleted_by)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return deleted
