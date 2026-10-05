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

        embedding = faces[0].embedding.astype(float).tolist()
        return GeneratedEmbedding(
            embedding=embedding,
            model_name=f"insightface/{settings.insightface_model_name}",
            face_count=len(faces),
        )

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


@lru_cache
def get_face_embedding_worker() -> FaceEmbeddingWorker:
    return FaceEmbeddingWorker()
