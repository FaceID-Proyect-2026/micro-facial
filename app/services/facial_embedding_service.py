from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.facial_embedding_repository import FacialEmbeddingRepository
from app.schemas.facial_embedding import (
    FacialEmbeddingCreate,
    FacialEmbeddingFromImageCreate,
    FacialEmbeddingPartialUpdate,
    FacialEmbeddingResponse,
    FacialEmbeddingSummary,
    FacialVerificationRequest,
    FacialVerificationResponse,
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

    async def verify_session_face(self, payload: FacialVerificationRequest) -> FacialVerificationResponse:
        generated = self.embedding_worker.generate(payload.image_base64)
        threshold = payload.threshold if payload.threshold is not None else settings.facial_match_threshold

        row = await self.repository.find_best_session_match(
            record_environment_id=payload.record_environment_id,
            embedding=generated.embedding,
        )
        if row is None:
            return FacialVerificationResponse(
                match=False,
                threshold=threshold,
                model_name=generated.model_name,
                reason="NO_ACTIVE_CANDIDATES",
            )

        similarity = float(row["similarity"])
        if similarity < threshold:
            return FacialVerificationResponse(
                match=False,
                id_apprentice=row["id_apprentice"],
                similarity=similarity,
                threshold=threshold,
                model_name=generated.model_name,
                reason="BELOW_THRESHOLD",
            )

        return FacialVerificationResponse(
            match=True,
            id_apprentice=row["id_apprentice"],
            similarity=similarity,
            threshold=threshold,
            model_name=generated.model_name,
            reason="MATCH_FOUND",
        )

    async def _register_embedding(self, payload: FacialEmbeddingCreate) -> FacialEmbeddingResponse:
        apprentice_id = await self._resolve_apprentice_id(payload.user_id)

        try:
            row = await self.repository.upsert(
                user_id=apprentice_id,
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
        apprentice_id = await self._resolve_apprentice_id(user_id)
        row = await self.repository.get_active_by_user(apprentice_id)
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

        apprentice_id = await self._resolve_apprentice_id(user_id)

        embedding = payload.embedding if "embedding" in payload.model_fields_set else None
        model_name = payload.model_name if "model_name" in payload.model_fields_set else None
        if payload.image_base64 is not None:
            generated = self.embedding_worker.generate(payload.image_base64)
            embedding = generated.embedding
            model_name = model_name or generated.model_name

        try:
            row = await self.repository.update_active(
                user_id=apprentice_id,
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
        apprentice_id = await self._resolve_apprentice_id(user_id)
        try:
            deleted = await self.repository.deactivate(apprentice_id, deleted_by)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return deleted

    async def _resolve_apprentice_id(self, user_id: UUID) -> UUID:
        apprentice_id = await self.repository.resolve_apprentice_id(user_id)
        if apprentice_id is None:
            return user_id
        return apprentice_id
