"""Pure facial liveness rules; independent of HTTP, SQL and model loading."""

import logging
import math

from app.core.config import settings

logger = logging.getLogger("facelit.embedding")


class LivenessValidator:
    REQUIRED_LIVENESS_CHALLENGES = 2

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
        if len(frames) < settings.liveness_min_frames:
            return False, "Se requieren suficientes frames para validar el reto."
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
            return False, "Debes realizar un reto específico de vida."
        if normalized == "BLINK":
            if self._has_mesh_metrics(frames):
                eye_min = self._metric_min(frames, "mesh_eye_ear")
                return (
                    self._metric_cycle(frames, "mesh_eye_ear", 0.045, -1) and eye_min < 0.23,
                    "El reto era pestañear.",
                )
            return False, "El reto era pestañear. No se pudieron medir los ojos; revisa la iluminación."
        if normalized == "NOD":
            if not self._has_mesh_metrics(frames):
                return False, "El reto era asentir con la cabeza."
            return self._metric_cycle(frames, "mesh_nose_y", 0.045), "El reto era asentir con la cabeza y volver al centro."
        if normalized == "SHAKE_HEAD":
            if not self._has_mesh_metrics(frames):
                return False, "El reto era negar con la cabeza."
            return self._metric_cycle(frames, "mesh_nose_x", 0.045), "El reto era negar con la cabeza y volver al centro."
        if normalized == "OPEN_CLOSE_MOUTH":
            if self._has_mesh_metrics(frames):
                mouth_max = self._metric_max(frames, "mesh_mouth_open_ratio")
                return (
                    self._metric_cycle(frames, "mesh_mouth_open_ratio", 0.055, 1) and mouth_max > 0.10,
                    "El reto era abrir y cerrar la boca.",
                )
            return False, "El reto era abrir y cerrar la boca. No se pudo medir la boca; revisa la iluminación."
        if normalized == "KISS_CAMERA":
            if not self._has_mesh_metrics(frames):
                return False, "El reto era dar un beso a la cámara."
            mouth_open = self._metric_max(frames, "mesh_mouth_open_ratio")
            return (
                self._metric_cycle(frames, "mesh_mouth_width_ratio", 0.018, -1)
                and mouth_open < 0.10
                and self._metric_delta(frames, "mesh_mouth_white_ratio") < 0.05,
                "El reto era dar un beso a la cámara, sin sonreír ni abrir la boca.",
            )
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
            return x_delta < -0.025, "El reto era mover el rostro hacia la izquierda."
        if normalized == "MOVE_RIGHT":
            return x_delta > 0.025, "El reto era mover el rostro hacia la derecha."
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

    def _metric_cycle(self, frames, key: str, threshold: float, direction: int = 0) -> bool:
        """Require neutral -> requested excursion -> neutral, not just any change."""
        values = self._metric_values(frames, key)
        if len(values) < 3:
            return False
        baseline = values[0]
        excursions = [(value - baseline) * direction if direction else abs(value - baseline)
                      for value in values[1:-1]]
        peak = max(excursions, default=0.0)
        return peak > threshold and abs(values[-1] - baseline) <= threshold / 2

    @staticmethod
    def _metric_values(frames, key: str) -> list[float]:
        return [float(frame.image_metrics.get(key, 0.0)) for frame in frames]

    def _metric_delta(self, frames, key: str) -> float:
        values = self._metric_values(frames, key)
        return max(values, default=0.0) - min(values, default=0.0)

    def _metric_max(self, frames, key: str) -> float:
        return max(self._metric_values(frames, key), default=0.0)

    def _metric_min(self, frames, key: str) -> float:
        values = self._metric_values(frames, key)
        return min(values, default=0.0)

    def _has_mesh_metrics(self, frames) -> bool:
        return all(float(frame.image_metrics.get("mesh_available", 0.0)) >= 1.0 for frame in frames)
