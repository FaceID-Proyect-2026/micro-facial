import base64
import binascii
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings


@dataclass(frozen=True)
class GeneratedEmbedding:
    embedding: list[float]
    model_name: str
    face_count: int
    landmarks: list[list[float]]
    bbox: list[float]
    image_metrics: dict[str, float]


class FaceEmbeddingWorker:
    def __init__(self) -> None:
        self._model = None

    def generate(self, image_base64: str) -> GeneratedEmbedding:
        image = decode_image_base64(image_base64)
        faces = self._get_model().get(image)

        if not faces:
            raise ValueError("No se detecto ningun rostro en la imagen.")
        if len(faces) > 1:
            raise ValueError("La imagen contiene mas de un rostro.")

        face = faces[0]
        embedding = face.embedding.astype(float).tolist()
        landmarks = getattr(face, "kps", None)
        bbox = getattr(face, "bbox", None)
        bbox_values = bbox.astype(float).tolist() if bbox is not None else []
        return GeneratedEmbedding(
            embedding=embedding,
            model_name=f"insightface/{settings.insightface_model_name}",
            face_count=len(faces),
            landmarks=landmarks.astype(float).tolist() if landmarks is not None else [],
            bbox=bbox_values,
            image_metrics=expression_metrics(image, bbox_values),
        )

    def warm_up(self) -> None:
        self._get_model()

    def _get_model(self):
        if self._model is None:
            matplotlib_cache = Path(".cache") / "matplotlib"
            matplotlib_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

            from insightface.app import FaceAnalysis

            model = FaceAnalysis(
                name=settings.insightface_model_name,
                providers=["CPUExecutionProvider"],
            )
            model.prepare(
                ctx_id=settings.insightface_ctx_id,
                det_size=(settings.insightface_det_size, settings.insightface_det_size),
            )
            self._model = model
        return self._model


def decode_image_base64(image_base64: str) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV y NumPy son requeridos para procesar imagenes. "
            "Instala las dependencias de requirements.txt."
        ) from exc

    if not image_base64:
        raise ValueError("La imagen es obligatoria.")

    _, _, payload = image_base64.partition(",")
    raw_payload = payload or image_base64

    try:
        image_bytes = base64.b64decode(raw_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("La imagen no tiene un formato base64 valido.") from exc

    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No fue posible leer la imagen enviada.")
    return image


def expression_metrics(image: Any, bbox: list[float]) -> dict[str, float]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {}

    if len(bbox) != 4:
        return {}

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(bbox[0])))
    y1 = max(0, min(height - 1, int(bbox[1])))
    x2 = max(0, min(width, int(bbox[2])))
    y2 = max(0, min(height, int(bbox[3])))
    face_width = max(x2 - x1, 1)
    face_height = max(y2 - y1, 1)

    def crop(rx1: float, ry1: float, rx2: float, ry2: float):
        ax1 = max(0, min(width - 1, x1 + int(face_width * rx1)))
        ay1 = max(0, min(height - 1, y1 + int(face_height * ry1)))
        ax2 = max(0, min(width, x1 + int(face_width * rx2)))
        ay2 = max(0, min(height, y1 + int(face_height * ry2)))
        if ax2 <= ax1 or ay2 <= ay1:
            return None
        return image[ay1:ay2, ax1:ax2]

    eye_region = crop(0.16, 0.20, 0.84, 0.48)
    mouth_region = crop(0.24, 0.56, 0.76, 0.88)

    def dark_ratio(region) -> float:
        if region is None or region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray < 75))

    def red_ratio(region) -> float:
        if region is None or region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        red = ((hue < 14) | (hue > 165)) & (saturation > 55) & (value > 70)
        return float(np.mean(red))

    return {
        "eye_dark_ratio": dark_ratio(eye_region),
        "mouth_dark_ratio": dark_ratio(mouth_region),
        "mouth_red_ratio": red_ratio(mouth_region),
    }


@lru_cache
def get_face_embedding_worker() -> FaceEmbeddingWorker:
    return FaceEmbeddingWorker()
