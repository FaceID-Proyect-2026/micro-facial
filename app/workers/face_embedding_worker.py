# Convierte imágenes en vectores y métricas faciales mediante el modelo de inferencia.
import base64
import binascii
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import Lock, local
from time import monotonic
from typing import Any

from app.core.config import settings
from app.domain.embedding import GeneratedEmbedding  # compatibility export for existing callers


class FaceEmbeddingWorker:
    def __init__(self) -> None:
        self._model = None
        self._inference_lock = Lock()
        self._inference_warmed = False
        self._recent_frames = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='facelit-inference')

    def generate(self, image_base64: str) -> GeneratedEmbedding:
        # Un solo hilo ejecuta las inferencias sobre el modelo compartido.
        return self._executor.submit(self._generate_serialized, image_base64).result()

    def _generate_serialized(self, image_base64: str) -> GeneratedEmbedding:
        # The cached MediaPipe/InsightFace objects are shared across HTTP requests.
        with self._inference_lock:
            now = monotonic()
            # Only derived inference features are retained briefly, never images,
            # account IDs, authorization or liveness decisions. The final request
            # must still validate both gestures and query the current database.
            for key, (expires, _) in list(self._recent_frames.items()):
                if expires <= now:
                    del self._recent_frames[key]
            enabled = settings.inference_cache_size > 0 and settings.inference_cache_ttl_seconds > 0
            key = sha256(image_base64.encode('utf-8')).digest() if enabled else None
            if key in self._recent_frames:
                expires, generated = self._recent_frames.pop(key)
                self._recent_frames[key] = (expires, generated)
                return generated
            generated = self._generate(image_base64)
            if enabled:
                self._recent_frames[key] = (monotonic() + settings.inference_cache_ttl_seconds, generated)
                while len(self._recent_frames) > settings.inference_cache_size:
                    self._recent_frames.popitem(last=False)
            return generated

    def _generate(self, image_base64: str) -> GeneratedEmbedding:
        image = decode_image_base64(image_base64)
        faces = self._get_model().get(image)

        if not faces:
            raise ValueError("No se detecto ningun rostro en la imagen.")
        if len(faces) > 1:
            raise ValueError("La imagen contiene mas de un rostro.")

        # El vector sirve para comparar rostros. Los puntos y métricas alimentan los retos de vida.
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
        self._executor.submit(self._warm_up_serialized).result()

    def _warm_up_serialized(self) -> None:
        with self._inference_lock:
            model = self._get_model()
            if not self._inference_warmed:
                self._prime_inference(model)
                self._inference_warmed = True

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _prime_inference(model) -> None:
        import numpy as np
        from insightface.data import get_image

        # Allocate/prepare inference kernels before serving the first capture.
        # Blank synthetic inputs are discarded; no user or biometric record is created.
        image = np.zeros((settings.insightface_det_size, settings.insightface_det_size, 3), dtype=np.uint8)
        model.get(image)
        recognition = model.models["recognition"]
        width, height = recognition.input_size
        recognition.get_feat(np.zeros((height, width, 3), dtype=np.uint8))
        face_mesh_metrics(image)
        # A blank frame does not run FaceMesh's landmark branch. Prime that
        # branch too, using the public sample bundled with pinned InsightFace,
        # before startup is considered complete. Never enroll or cache it.
        sample = get_image("t1")
        if sample.shape[:2] != (886, 1280):
            raise RuntimeError("The bundled inference warm-up sample changed; review its crop")
        face_mesh_metrics(sample[180:450, 410:650])

    def _get_model(self):
        if self._model is None:
            matplotlib_cache = Path(".cache") / "matplotlib"
            matplotlib_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

            model = load_face_analysis(
                name=settings.insightface_model_name,
                # kps/bbox come from detection; expression metrics use MediaPipe below.
                # Age/gender and InsightFace's additional landmark models are unused.
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            model.prepare(
                ctx_id=settings.insightface_ctx_id,
                det_size=(settings.insightface_det_size, settings.insightface_det_size),
            )
            self._model = model
        return self._model


def load_face_analysis(**options):
    """Keep InsightFace inference methods, avoiding unused sessions in its pinned pack."""
    from insightface.app import FaceAnalysis

    if options['name'] != 'buffalo_l':
        return FaceAnalysis(**options)

    from insightface import model_zoo
    from insightface.utils.storage import ensure_available

    # InsightFace 0.7.3 initializes every ONNX session before filtering modules.
    # These are the same two files used by its buffalo_l detection/recognition;
    # no weights, thresholds, preprocessing, get() or prepare() are replaced.
    model_dir = ensure_available('models', 'buffalo_l', root='~/.insightface')
    analysis = FaceAnalysis.__new__(FaceAnalysis)
    analysis.model_dir = model_dir
    analysis.models = {}
    for task, filename in (('detection', 'det_10g.onnx'), ('recognition', 'w600k_r50.onnx')):
        path = Path(model_dir) / filename
        if not path.is_file():
            raise RuntimeError(f'Missing required buffalo_l model: {filename}')
        model = model_zoo.get_model(str(path), providers=options['providers'])
        if model is None or model.taskname != task:
            raise RuntimeError(f'Unexpected model task in buffalo_l: {filename}')
        analysis.models[task] = model
    analysis.det_model = analysis.models['detection']
    return analysis


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
    face_height = max(distance(10, 152), 1.0)
    face_width = max(distance(234, 454), 1.0)
    left_brow_gap = ((point(105)[1] + point(66)[1]) / 2 - (point(159)[1] + point(145)[1]) / 2) / face_height
    right_brow_gap = ((point(334)[1] + point(296)[1]) / 2 - (point(386)[1] + point(374)[1]) / 2) / face_height
    # Relative to the face, so translating the whole head is not mistaken for rotation.
    nose_x = (point(1)[0] - (point(234)[0] + point(454)[0]) / 2) / face_width
    nose_y = (point(1)[1] - point(10)[1]) / face_height
    face_width_ratio = face_width / max(float(width), 1.0)
    mouth_red_ratio = mesh_mouth_red_ratio(image, landmarks)
    mouth_white_ratio = mesh_mouth_white_ratio(image, landmarks)

    return {
        "mesh_available": 1.0,
        "mesh_eye_ear": float((left_eye_ear + right_eye_ear) / 2.0),
        "mesh_left_eye_ear": float(left_eye_ear),
        "mesh_right_eye_ear": float(right_eye_ear),
        "mesh_mouth_open_ratio": float(mouth_open_ratio),
        "mesh_mouth_width_ratio": float(mouth_width / face_width),
        "mesh_mouth_red_ratio": float(mouth_red_ratio),
        "mesh_mouth_white_ratio": float(mouth_white_ratio),
        "mesh_left_brow_gap": float(left_brow_gap),
        "mesh_right_brow_gap": float(right_brow_gap),
        "mesh_nose_x": float(nose_x),
        "mesh_nose_y": float(nose_y),
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


def mesh_mouth_white_ratio(image: Any, landmarks: Any) -> float:
    """Estimate visible teeth using bright, low-saturation pixels in the mouth ROI."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.0

    height, width = image.shape[:2]
    xs = [landmarks[index].x * width for index in (61, 291, 13, 14)]
    ys = [landmarks[index].y * height for index in (61, 291, 13, 14)]
    padding_x = max(4, int((max(xs) - min(xs)) * 0.18))
    padding_y = max(4, int((max(ys) - min(ys)) * 0.35))
    x1 = max(0, int(min(xs)) - padding_x)
    y1 = max(0, int(min(ys)) - padding_y)
    x2 = min(width, int(max(xs)) + padding_x)
    y2 = min(height, int(max(ys)) + padding_y)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 145)
    return float(np.mean(white))


_face_mesh_state = local()


def get_face_mesh():
    cached = getattr(_face_mesh_state, 'model', None)
    if cached is not None:
        return cached
    matplotlib_cache = Path(".cache") / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

    try:
        import mediapipe as mp
    except ImportError:
        return None

    model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    _face_mesh_state.model = model
    return model


@lru_cache
def get_face_embedding_worker() -> FaceEmbeddingWorker:
    return FaceEmbeddingWorker()


def close_face_embedding_worker() -> None:
    if get_face_embedding_worker.cache_info().currsize:
        worker = get_face_embedding_worker()
        get_face_embedding_worker.cache_clear()
        worker.close()
