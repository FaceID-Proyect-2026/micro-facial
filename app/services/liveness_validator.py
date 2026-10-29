"""Pure facial liveness rules; independent of HTTP, SQL and model loading."""

import logging
import math

from app.core.config import settings

logger = logging.getLogger("facelit.embedding")


class LivenessValidator:
    REQUIRED_LIVENESS_CHALLENGES = 2
    SUPPORTED_CHALLENGES = frozenset({"MOVE_UP", "MOVE_DOWN", "MOVE_CLOSER", "MOVE_AWAY"})
    REQUIRED_METRICS = {
        "MOVE_UP": ("mesh_nose_y",),
        "MOVE_DOWN": ("mesh_nose_y",),
    }

    def validate_sequence(self, frames, challenges: list[str]) -> tuple[bool, str]:
        if len(challenges) != self.REQUIRED_LIVENESS_CHALLENGES or len(set(challenges)) != self.REQUIRED_LIVENESS_CHALLENGES:
            return False, "Se requieren dos retos de vida diferentes y completos."

        required_frames = settings.liveness_min_frames * len(challenges)
        if len(frames) < required_frames:
            return False, f"La secuencia requiere {len(challenges)} retos completos."
        if len(frames) % len(challenges):
            return False, "Cada reto debe contener el mismo número de frames."

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
            live, reason = self.validate(chunk, challenge)
            if not live:
                return False, f"Reto {index + 1}/{len(challenges)} fallido: {reason}"

        return True, "LIVE_OK_SEQUENCE"

    def validate(self, frames, challenge: str | None = None) -> tuple[bool, str]:
        normalized = (challenge or "").strip().upper()
        if normalized not in self.SUPPORTED_CHALLENGES:
            return False, "El reto solicitado no está habilitado; inicia una nueva captura."
        if len(frames) < settings.liveness_min_frames:
            return False, "Se requieren suficientes frames para validar el reto."
        if any(not frame.landmarks or len(frame.landmarks) < 2 for frame in frames):
            return False, "No fue posible medir puntos faciales en toda la secuencia."

        required = self.REQUIRED_METRICS.get((challenge or "").strip().upper())
        if required and not self._has_mesh_metrics(frames, *required):
            return False, "No se pudo medir el gesto en todas las imágenes; revisa la iluminación y mantén el rostro visible."

        reference = frames[0]
        if any(self._cosine_similarity(reference.embedding, frame.embedding) < settings.liveness_min_embedding_similarity for frame in frames[1:]):
            return False, "La secuencia no corresponde de forma consistente a la misma persona."

        max_landmark_motion = max(self._normalized_landmark_motion(reference, frame) for frame in frames[1:])
        max_box_motion = max(self._normalized_box_motion(reference, frame) for frame in frames[1:])
        challenge_ok, challenge_reason = self._validate_challenge(frames, challenge)
        logger.info(
            "liveness metrics: landmark_motion=%.5f box_motion=%.5f nose_delta=%.5f challenge=%s accepted=%s reason=%s",
            max_landmark_motion, max_box_motion, self._metric_delta(frames, "mesh_nose_y"),
            challenge, challenge_ok, challenge_reason,
        )
        if not challenge_ok:
            return False, challenge_reason

        return True, "LIVE_OK"

    @staticmethod
    def resolve_challenges(challenges: list[str] | None, challenge: str | None) -> list[str]:
        resolved = [item.strip().upper() for item in (challenges or []) if item and item.strip()]
        if resolved:
            return resolved
        if challenge and "," in challenge:
            return [item.strip().upper() for item in challenge.split(",") if item.strip()]
        if challenge and challenge.strip():
            return [challenge.strip().upper()]
        return []

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
            return False, "Debes realizar un reto específico de vida."
        first = frames[0]
        last = frames[-1]
        if len(first.bbox) != 4 or len(last.bbox) != 4:
            return False, "No fue posible validar el reto solicitado."

        first_width = max(first.bbox[2] - first.bbox[0], 1.0)
        first_height = max(first.bbox[3] - first.bbox[1], 1.0)
        last_width = max(last.bbox[2] - last.bbox[0], 1.0)
        last_height = max(last.bbox[3] - last.bbox[1], 1.0)
        area_delta = ((last_width * last_height) - (first_width * first_height)) / max(first_width * first_height, 1.0)

        if normalized == "MOVE_UP":
            if self._has_mesh_metrics(frames):
                values = self._metric_values(frames, "mesh_nose_y")
                return values[0] - values[-1] > 0.045, "El reto era girar el rostro hacia arriba."
            return False, "El reto era girar el rostro hacia arriba. No se pudo medir la orientación."
        if normalized == "MOVE_DOWN":
            if self._has_mesh_metrics(frames):
                values = self._metric_values(frames, "mesh_nose_y")
                return values[-1] - values[0] > 0.045, "El reto era girar el rostro hacia abajo."
            return False, "El reto era girar el rostro hacia abajo. No se pudo medir la orientación."
        if normalized == "MOVE_CLOSER":
            return area_delta > 0.015, "El reto era acercarse un poco a la camara."
        if normalized == "MOVE_AWAY":
            return area_delta < -0.015, "El reto era alejarse un poco de la camara."
        return False, "El reto solicitado no está soportado; inicia una nueva captura."

    @staticmethod
    def _metric_values(frames, key: str) -> list[float]:
        return [float(frame.image_metrics.get(key, 0.0)) for frame in frames]

    def _metric_delta(self, frames, key: str) -> float:
        values = self._metric_values(frames, key)
        return max(values, default=0.0) - min(values, default=0.0)

    def _has_mesh_metrics(self, frames, *required: str) -> bool:
        for frame in frames:
            try:
                values = [float(frame.image_metrics[key]) for key in ("mesh_available", *required)]
            except (KeyError, TypeError, ValueError):
                return False
            if values[0] < 1.0 or not all(math.isfinite(value) for value in values):
                return False
        return True
