import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.facial_embedding_repository import FacialEmbeddingRepository
from app.services.liveness_validator import LivenessValidator
from app.services.ports import EmbeddingGenerator, FacialEmbeddingStore, LivenessPolicy
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
from app.workers.face_embedding_worker import get_face_embedding_worker

logger = logging.getLogger("facelit.embedding")


class DuplicateFaceError(ValueError):
    """Raised when a face is already enrolled in another apprentice account."""

    message = "este rostro ya fue registrado anteriormente en otra cuenta"

    def __init__(self) -> None:
        super().__init__(self.message)


class FacialEmbeddingService:
    """Application facade: coordinates facial rules, model generation and persistence."""

    REQUIRED_LIVENESS_CHALLENGES = LivenessValidator.REQUIRED_LIVENESS_CHALLENGES
    def __init__(
        self,
        session: AsyncSession,
        embedding_worker: EmbeddingGenerator | None = None,
        *,
        repository: FacialEmbeddingStore | None = None,
        liveness_validator: LivenessPolicy | None = None,
    ) -> None:
        self.session = session
        self.repository = repository if repository is not None else FacialEmbeddingRepository(session)
        self.liveness_validator = liveness_validator if liveness_validator is not None else LivenessValidator()
        self.embedding_worker = embedding_worker or get_face_embedding_worker()

    async def register(self, payload: FacialEmbeddingCreate) -> FacialEmbeddingResponse:
        if payload.reset_token is not None:
            raise ValueError("El re-registro autorizado debe realizarse con captura en vivo desde from-image.")
        return await self._register_embedding(payload)

    async def register_from_image(self, payload: FacialEmbeddingFromImageCreate) -> FacialEmbeddingResponse:
        generated = self._generate_registration_embedding(payload)
        embedding_payload = FacialEmbeddingCreate(
            user_id=payload.user_id,
            embedding=generated.embedding,
            model_name=generated.model_name,
            photo_reference=payload.photo_reference,
            replace_existing=payload.replace_existing,
            reset_token=payload.reset_token,
            created_by=payload.created_by,
        )
        return await self._register_embedding(embedding_payload)

    def _generate_registration_embedding(self, payload: FacialEmbeddingFromImageCreate):
        frames = payload.image_frames[: settings.liveness_max_frames]
        challenges = self._resolve_liveness_challenges(payload.liveness_challenges, payload.liveness_challenge)
        if len(frames) < settings.liveness_min_frames * self.REQUIRED_LIVENESS_CHALLENGES:
            raise ValueError("El registro requiere dos retos de vida completos capturados en cámara.")

        generated_frames = [self.embedding_worker.generate(frame) for frame in frames]
        live, liveness_reason = self._validate_liveness_sequence(generated_frames, challenges)
        if not live:
            raise ValueError(liveness_reason)
        logger.info(
            "from-image liveness accepted: user=%s frames=%s challenges=%s reason=%s",
            payload.user_id,
            len(frames),
            challenges,
            liveness_reason,
        )
        return generated_frames[-1]

    async def verify_session_face(self, payload: FacialVerificationRequest) -> FacialVerificationResponse:
        threshold = payload.threshold if payload.threshold is not None else settings.facial_match_threshold
        frames = payload.image_frames[: settings.liveness_max_frames]
        challenges = self._resolve_liveness_challenges(payload.liveness_challenges, payload.liveness_challenge)
        if len(frames) < settings.liveness_min_frames:
            logger.info(
                "verify-session rejected: reason=LIVENESS_REQUIRED frames=%s challenges=%s",
                len(frames),
                challenges,
            )
            return FacialVerificationResponse(
                match=False,
                live=False,
                threshold=threshold,
                model_name=f"insightface/{settings.insightface_model_name}",
                reason="LIVENESS_REQUIRED",
                liveness_reason="Se requiere una secuencia de frames capturada en vivo.",
            )

        generated_frames = [self.embedding_worker.generate(frame) for frame in frames]
        live, liveness_reason = self._validate_liveness_sequence(generated_frames, challenges)
        generated = generated_frames[-1]
        if not live:
            logger.info(
                "verify-session rejected: reason=LIVENESS_FAILED liveness_reason=%s frames=%s challenges=%s",
                liveness_reason,
                len(frames),
                challenges,
            )
            return FacialVerificationResponse(
                match=False,
                live=False,
                threshold=threshold,
                model_name=generated.model_name,
                reason="LIVENESS_FAILED",
                liveness_reason=liveness_reason,
            )

        row = await self.repository.find_best_session_match(
            record_environment_id=payload.record_environment_id,
            embedding=generated.embedding,
        )
        if row is None:
            logger.info(
                "verify-session rejected: reason=NO_ACTIVE_CANDIDATES live=true frames=%s challenges=%s",
                len(frames),
                challenges,
            )
            return FacialVerificationResponse(
                match=False,
                live=True,
                threshold=threshold,
                model_name=generated.model_name,
                reason="NO_ACTIVE_CANDIDATES",
                liveness_reason="LIVE_OK",
            )

        similarity = float(row["similarity"])
        if similarity < threshold:
            logger.info(
                "verify-session rejected: reason=BELOW_THRESHOLD live=true apprentice=%s similarity=%.4f threshold=%.4f challenges=%s",
                row["id_apprentice"],
                similarity,
                threshold,
                challenges,
            )
            return FacialVerificationResponse(
                match=False,
                live=True,
                id_apprentice=row["id_apprentice"],
                similarity=similarity,
                threshold=threshold,
                model_name=generated.model_name,
                reason="BELOW_THRESHOLD",
                liveness_reason="LIVE_OK",
            )

        logger.info(
            "verify-session accepted: reason=MATCH_FOUND live=true apprentice=%s similarity=%.4f threshold=%.4f challenges=%s",
            row["id_apprentice"],
            similarity,
            threshold,
            challenges,
        )
        return FacialVerificationResponse(
            match=True,
            live=True,
            id_apprentice=row["id_apprentice"],
            similarity=similarity,
            threshold=threshold,
            model_name=generated.model_name,
            reason="MATCH_FOUND",
            liveness_reason="LIVE_OK",
        )

    def check_liveness(self, payload: FacialLivenessCheckRequest) -> FacialLivenessCheckResponse:
        frames = payload.image_frames[: settings.liveness_max_frames]
        if len(frames) < settings.liveness_min_frames:
            return FacialLivenessCheckResponse(
                live=False,
                reason="La cámara no capturó suficientes imágenes para validar el reto.",
            )

        generated_frames = [self.embedding_worker.generate(frame) for frame in frames]
        live, reason = self._validate_liveness(generated_frames, payload.liveness_challenge)
        return FacialLivenessCheckResponse(live=live, reason=reason)

    # Compatibility for callers that already exercise these rules through the facade.
    def _validate_liveness_sequence(self, frames, challenges: list[str]) -> tuple[bool, str]:
        return self.liveness_validator.validate_sequence(frames, challenges)

    def _validate_liveness(self, frames, challenge: str | None = None) -> tuple[bool, str]:
        return self.liveness_validator.validate(frames, challenge)

    @staticmethod
    def _resolve_liveness_challenges(challenges: list[str] | None, challenge: str | None) -> list[str]:
        return LivenessValidator.resolve_challenges(challenges, challenge)

    async def _register_embedding(self, payload: FacialEmbeddingCreate) -> FacialEmbeddingResponse:
        apprentice_id = await self._resolve_apprentice_id(payload.user_id)

        try:
            duplicate = await self.repository.find_duplicate_embedding(
                apprentice_id=apprentice_id,
                embedding=payload.embedding,
                threshold=settings.facial_match_threshold,
            )
            if duplicate is not None:
                logger.warning(
                    "facial registration rejected: duplicate face apprentice=%s existing_apprentice=%s similarity=%.4f",
                    apprentice_id,
                    duplicate["id_apprentice"],
                    float(duplicate["similarity"]),
                )
                raise DuplicateFaceError()
            await self.repository.authorize_registration(apprentice_id, payload.replace_existing, payload.reset_token)
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
        if payload.embedding is not None or payload.image_base64 is not None:
            raise ValueError("Para reemplazar el rostro solicita autorización y usa la captura en vivo de from-image.")
        editable_fields = {"embedding", "image_base64", "model_name", "photo_reference"}
        if not payload.model_fields_set.intersection(editable_fields):
            raise ValueError("Debe enviar al menos un campo para actualizar.")
        if "embedding" in payload.model_fields_set and payload.embedding is None:
            raise ValueError("El embedding no puede ser nulo.")
        if "image_base64" in payload.model_fields_set and payload.image_base64 is None:
            raise ValueError("La imagen no puede ser nula.")
        if "model_name" in payload.model_fields_set and payload.model_name is None:
            raise ValueError("model_name no puede ser nulo.")

        apprentice_id = await self._resolve_apprentice_id(user_id)

        model_name = payload.model_name if "model_name" in payload.model_fields_set else None

        try:
            row = await self.repository.update_active(
                user_id=apprentice_id,
                embedding=None,
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
