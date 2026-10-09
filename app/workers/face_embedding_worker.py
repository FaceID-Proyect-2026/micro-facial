import base64
import binascii
import math
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

    metrics = {
        "eye_dark_ratio": dark_ratio(eye_region),
        "mouth_dark_ratio": dark_ratio(mouth_region),
        "mouth_red_ratio": red_ratio(mouth_region),
    }
    metrics.update(face_mesh_metrics(image))
    return metrics


def face_mesh_metrics(image: Any) -> dict[str, float]:
    face_mesh = get_face_mesh()
    if face_mesh is None:
        return {}

    try:
        import cv2
        import numpy as np
    except ImportError:
        return {}

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks or len(result.multi_face_landmarks) != 1:
        return {}

    height, width = image.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark

    def point(index: int) -> tuple[float, float]:
        landmark = landmarks[index]
        return landmark.x * width, landmark.y * height

    def distance(a: int, b: int) -> float:
        return math.dist(point(a), point(b))

    def eye_aspect_ratio(indices: tuple[int, int, int, int, int, int]) -> float:
        p1, p2, p3, p4, p5, p6 = indices
        horizontal = distance(p1, p4)
        if horizontal <= 0:
            return 0.0
        return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)

    left_eye_ear = eye_aspect_ratio((33, 160, 158, 133, 153, 144))
    right_eye_ear = eye_aspect_ratio((362, 385, 387, 263, 373, 380))
    mouth_width = max(distance(61, 291), 1.0)
    mouth_open_ratio = distance(13, 14) / mouth_width
    nose_x = landmarks[1].x
    face_width_ratio = distance(234, 454) / max(float(width), 1.0)
    mouth_red_ratio = mesh_mouth_red_ratio(image, landmarks)

    return {
        "mesh_available": 1.0,
        "mesh_eye_ear": float((left_eye_ear + right_eye_ear) / 2.0),
        "mesh_left_eye_ear": float(left_eye_ear),
        "mesh_right_eye_ear": float(right_eye_ear),
        "mesh_mouth_open_ratio": float(mouth_open_ratio),
        "mesh_mouth_red_ratio": float(mouth_red_ratio),
        "mesh_nose_x": float(nose_x),
        "mesh_face_width_ratio": float(face_width_ratio),
    }


def mesh_mouth_red_ratio(image: Any, landmarks: Any) -> float:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.0

    height, width = image.shape[:2]
    mouth_indices = (0, 13, 14, 17, 37, 39, 40, 61, 78, 81, 82, 87, 88, 178, 181, 191, 267, 269, 270, 291, 308, 311, 312, 317, 318, 402, 405, 415)
    xs = [landmarks[index].x * width for index in mouth_indices]
    ys = [landmarks[index].y * height for index in mouth_indices]
    padding_x = max(4, int((max(xs) - min(xs)) * 0.18))
    padding_y = max(4, int((max(ys) - min(ys)) * 0.55))
    x1 = max(0, int(min(xs)) - padding_x)
    y1 = max(0, int(min(ys)) - padding_y)
    x2 = min(width, int(max(xs)) + padding_x)
    y2 = min(height, int(max(ys)) + padding_y)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red_or_pink = ((hue < 16) | (hue > 160)) & (saturation > 45) & (value > 55)
    return float(np.mean(red_or_pink))


@lru_cache
def get_face_mesh():
    matplotlib_cache = Path(".cache") / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

    try:
        import mediapipe as mp
    except ImportError:
        return None

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


@lru_cache
def get_face_embedding_worker() -> FaceEmbeddingWorker:
    return FaceEmbeddingWorker()
