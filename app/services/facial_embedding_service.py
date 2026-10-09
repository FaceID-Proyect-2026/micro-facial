import logging
import math
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
    FacialLivenessCheckRequest,
    FacialLivenessCheckResponse,
    FacialVerificationRequest,
    FacialVerificationResponse,
)
from app.workers.face_embedding_worker import FaceEmbeddingWorker, get_face_embedding_worker

logger = logging.getLogger("facelit.embedding")


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
        generated = self._generate_registration_embedding(payload)
        embedding_payload = FacialEmbeddingCreate(
            user_id=payload.user_id,
            embedding=generated.embedding,
            model_name=generated.model_name,
            photo_reference=payload.photo_reference,
            replace_existing=payload.replace_existing,
            created_by=payload.created_by,
        )
        return await self._register_embedding(embedding_payload)

    def _generate_registration_embedding(self, payload: FacialEmbeddingFromImageCreate):
        frames = payload.image_frames[: settings.liveness_max_frames]
        challenges = self._resolve_liveness_challenges(payload.liveness_challenges, payload.liveness_challenge)
        if not frames:
            return self.embedding_worker.generate(payload.image_base64)
        if len(frames) < settings.liveness_min_frames:
            raise ValueError("Se requiere una secuencia de frames capturada en vivo.")

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

    def _validate_liveness_sequence(self, frames, challenges: list[str]) -> tuple[bool, str]:
        if not challenges:
            return self._validate_liveness(frames)

        if len(challenges) == 1:
            return self._validate_liveness(frames, challenges[0])

        required_frames = settings.liveness_min_frames * len(challenges)
        if len(frames) < required_frames:
            return False, f"La secuencia requiere {len(challenges)} retos completos."

        reference = frames[0]
        if any(self._cosine_similarity(reference.embedding, frame.embedding) < settings.liveness_min_embedding_similarity for frame in frames[1:]):
            return False, "La secuencia no corresponde de forma consistente a la misma persona."

        frames_per_challenge = len(frames) // len(challenges)
        if frames_per_challenge < settings.liveness_min_frames:
            return False, "Cada reto requiere suficientes frames capturados en vivo."

        for index, challenge in enumerate(challenges):
            start = index * frames_per_challenge
            end = start + frames_per_challenge
            if index == len(challenges) - 1:
                end = len(frames)
            chunk = frames[start:end]
            live, reason = self._validate_liveness(chunk, challenge)
            if not live:
                return False, f"Reto {index + 1}/{len(challenges)} fallido: {reason}"

        return True, "LIVE_OK_SEQUENCE"

    def _validate_liveness(self, frames, challenge: str | None = None) -> tuple[bool, str]:
        if any(not frame.landmarks or len(frame.landmarks) < 2 for frame in frames):
            return False, "No fue posible medir puntos faciales en toda la secuencia."

        reference = frames[0]
        if any(self._cosine_similarity(reference.embedding, frame.embedding) < settings.liveness_min_embedding_similarity for frame in frames[1:]):
            return False, "La secuencia no corresponde de forma consistente a la misma persona."

        max_landmark_motion = max(self._normalized_landmark_motion(reference, frame) for frame in frames[1:])
        max_box_motion = max(self._normalized_box_motion(reference, frame) for frame in frames[1:])
        challenge_ok, challenge_reason = self._validate_challenge(frames, challenge)
        logger.info(
            "liveness metrics: landmark_motion=%.5f box_motion=%.5f eye_dark_delta=%.5f mouth_dark_delta=%.5f mouth_red_max=%.5f mesh_eye_delta=%.5f mesh_eye_min=%.5f mesh_mouth_delta=%.5f mesh_mouth_max=%.5f mesh_tongue_max=%.5f challenge=%s challenge_ok=%s challenge_reason=%s",
            max_landmark_motion,
            max_box_motion,
            self._metric_delta(frames, "eye_dark_ratio"),
            self._metric_delta(frames, "mouth_dark_ratio"),
            self._metric_max(frames, "mouth_red_ratio"),
            self._metric_delta(frames, "mesh_eye_ear"),
            self._metric_min(frames, "mesh_eye_ear"),
            self._metric_delta(frames, "mesh_mouth_open_ratio"),
            self._metric_max(frames, "mesh_mouth_open_ratio"),
            self._metric_max(frames, "mesh_mouth_red_ratio"),
            challenge,
            challenge_ok,
            challenge_reason,
        )
        if (
            max_landmark_motion < settings.liveness_min_landmark_motion
            and max_box_motion < settings.liveness_min_box_motion
            and self._is_generic_challenge(challenge)
        ):
            return False, "No se detecto movimiento facial suficiente; evita usar fotos o pantallas."
        if not challenge_ok:
            return False, challenge_reason

        return True, "LIVE_OK"

    @staticmethod
    def _resolve_liveness_challenges(challenges: list[str] | None, challenge: str | None) -> list[str]:
        resolved = [item.strip().upper() for item in (challenges or []) if item and item.strip()]
        if resolved:
            return resolved[:3]
        if challenge and "," in challenge:
            return [item.strip().upper() for item in challenge.split(",") if item.strip()][:3]
        if challenge and challenge.strip():
            return [challenge.strip().upper()]
        return []

    @staticmethod
    def _is_generic_challenge(challenge: str | None) -> bool:
        normalized = (challenge or "ANY_MOVEMENT").strip().upper()
        return normalized in {"", "ANY_MOVEMENT"}

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _normalized_landmark_motion(left, right) -> float:
        pairs = zip(left.landmarks, right.landmarks)
        distance = sum(math.dist(a, b) for a, b in pairs) / max(1, min(len(left.landmarks), len(right.landmarks)))
        box = left.bbox if len(left.bbox) == 4 else right.bbox
        if len(box) != 4:
            return 0.0
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        if diagonal <= 0:
            return 0.0
        return distance / diagonal

    @staticmethod
    def _normalized_box_motion(left, right) -> float:
        if len(left.bbox) != 4 or len(right.bbox) != 4:
            return 0.0

        left_width = max(left.bbox[2] - left.bbox[0], 1.0)
        left_height = max(left.bbox[3] - left.bbox[1], 1.0)
        right_width = max(right.bbox[2] - right.bbox[0], 1.0)
        right_height = max(right.bbox[3] - right.bbox[1], 1.0)
        diagonal = math.hypot(left_width, left_height)
        if diagonal <= 0:
            return 0.0

        left_center = ((left.bbox[0] + left.bbox[2]) / 2, (left.bbox[1] + left.bbox[3]) / 2)
        right_center = ((right.bbox[0] + right.bbox[2]) / 2, (right.bbox[1] + right.bbox[3]) / 2)
        center_motion = math.dist(left_center, right_center) / diagonal
        scale_motion = abs((right_width * right_height) - (left_width * left_height)) / max(left_width * left_height, 1.0)
        return center_motion + min(scale_motion, 0.5)

    def _validate_challenge(self, frames, challenge: str | None) -> tuple[bool, str]:
        normalized = (challenge or "ANY_MOVEMENT").strip().upper()
        if normalized in {"", "ANY_MOVEMENT"}:
            return True, "LIVE_OK"
        if normalized == "BLINK":
            if self._has_mesh_metrics(frames):
                eye_delta = self._metric_delta(frames, "mesh_eye_ear")
                eye_min = self._metric_min(frames, "mesh_eye_ear")
                return (
                    (eye_delta > 0.045 and eye_min < 0.23) or eye_delta > 0.075,
                    "El reto era pestañear.",
                )
            return self._metric_delta(frames, "eye_dark_ratio") > 0.018, "El reto era pestañear."
        if normalized == "OPEN_CLOSE_MOUTH":
            if self._has_mesh_metrics(frames):
                mouth_delta = self._metric_delta(frames, "mesh_mouth_open_ratio")
                mouth_max = self._metric_max(frames, "mesh_mouth_open_ratio")
                return (
                    mouth_delta > 0.055 and mouth_max > 0.10,
                    "El reto era abrir y cerrar la boca.",
                )
            return self._metric_delta(frames, "mouth_dark_ratio") > 0.030, "El reto era abrir y cerrar la boca."
        if normalized == "STICK_TONGUE":
            if self._has_mesh_metrics(frames):
                red_max = self._metric_max(frames, "mesh_mouth_red_ratio")
                red_delta = self._metric_delta(frames, "mesh_mouth_red_ratio")
                mouth_max = self._metric_max(frames, "mesh_mouth_open_ratio")
                return (
                    (red_max > 0.060 and mouth_max > 0.075) or red_delta > 0.030,
                    "El reto era sacar la lengua.",
                )
            return self._metric_max(frames, "mouth_red_ratio") > 0.018, "El reto era sacar la lengua."

        first = frames[0]
        last = frames[-1]
        if len(first.bbox) != 4 or len(last.bbox) != 4:
            return False, "No fue posible validar el reto solicitado."

        first_width = max(first.bbox[2] - first.bbox[0], 1.0)
        first_height = max(first.bbox[3] - first.bbox[1], 1.0)
        last_width = max(last.bbox[2] - last.bbox[0], 1.0)
        last_height = max(last.bbox[3] - last.bbox[1], 1.0)
        diagonal = math.hypot(first_width, first_height)
        first_center_x = (first.bbox[0] + first.bbox[2]) / 2
        last_center_x = (last.bbox[0] + last.bbox[2]) / 2
        x_delta = (last_center_x - first_center_x) / diagonal
        area_delta = ((last_width * last_height) - (first_width * first_height)) / max(first_width * first_height, 1.0)

        if normalized == "MOVE_LEFT":
            return x_delta < -0.006 or x_delta > 0.006, "El reto era mover el rostro hacia la izquierda."
        if normalized == "MOVE_RIGHT":
            return x_delta > 0.006 or x_delta < -0.006, "El reto era mover el rostro hacia la derecha."
        if normalized == "MOVE_CLOSER":
            return area_delta > 0.015, "El reto era acercarse un poco a la camara."
        if normalized == "MOVE_AWAY":
            return area_delta < -0.015, "El reto era alejarse un poco de la camara."
        return True, "LIVE_OK"

    @staticmethod
    def _metric_values(frames, key: str) -> list[float]:
        return [float(frame.image_metrics.get(key, 0.0)) for frame in frames]

    def _metric_delta(self, frames, key: str) -> float:
        values = self._metric_values(frames, key)
        return max(values, default=0.0) - min(values, default=0.0)

    def _metric_max(self, frames, key: str) -> float:
        return max(self._metric_values(frames, key), default=0.0)

    def _metric_min(self, frames, key: str) -> float:
        values = [value for value in self._metric_values(frames, key) if value > 0.0]
        return min(values, default=0.0)

    def _has_mesh_metrics(self, frames) -> bool:
        return all(float(frame.image_metrics.get("mesh_available", 0.0)) >= 1.0 for frame in frames)

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
