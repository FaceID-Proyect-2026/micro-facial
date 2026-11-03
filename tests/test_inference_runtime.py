"""Runtime isolation and responsiveness; gesture acceptance lives in contract tests."""

import asyncio
import sys
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx

from app.api.dependencies import facial_service
from app.core.config import Settings, settings
from app.main import create_app
from app.schemas.facial_embedding import FacialEmbeddingFromImageCreate, FacialLivenessCheckResponse
from app.services.facial_embedding_service import FacialEmbeddingService
from app.workers.face_embedding_worker import FaceEmbeddingWorker
from tests.support.liveness_signals import frames_for


def test_shared_dotenv_accepts_compose_published_port(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SERVICE_PORT=8091\n", encoding="utf-8")
    assert Settings(_env_file=env_file, database_url="postgresql+asyncpg://test:unused@127.0.0.1/test").service_port == 8091


def test_model_loads_required_tasks_once_without_changing_detection_settings(monkeypatch, tmp_path):
    instances = []

    class Model:
        def __init__(self, **options):
            self.options = options
            instances.append(self)

        def prepare(self, **options):
            self.prepared = options

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('app.workers.face_embedding_worker.load_face_analysis', lambda **options: Model(**options))
    worker = FaceEmbeddingWorker()
    primed = []
    monkeypatch.setattr(worker, "_prime_inference", lambda model: primed.append(model))
    worker.warm_up()
    worker.warm_up()

    assert len(instances) == 1
    assert primed == instances
    assert instances[0].options == {
        "name": settings.insightface_model_name,
        "allowed_modules": ["detection", "recognition"],
        "providers": ["CPUExecutionProvider"],
    }
    assert instances[0].prepared == {
        "ctx_id": settings.insightface_ctx_id,
        "det_size": (settings.insightface_det_size, settings.insightface_det_size),
    }


def test_warmup_primes_the_real_landmark_branch_without_enrolling(monkeypatch):
    from app.workers import face_embedding_worker as module

    class Image:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, crop):
            return Image((270, 240, 3))

    meshes = []
    recognitions = []
    detections = []
    monkeypatch.setitem(sys.modules, 'numpy', SimpleNamespace(zeros=lambda shape, dtype: Image(shape), uint8='uint8'))
    monkeypatch.setitem(sys.modules, 'insightface.data', SimpleNamespace(get_image=lambda name: Image((886, 1280, 3))))
    monkeypatch.setattr(module, 'face_mesh_metrics', lambda image: meshes.append(image.shape))
    model = SimpleNamespace(
        get=lambda image: detections.append(image.shape),
        models={'recognition': SimpleNamespace(input_size=(112,112), get_feat=lambda image: recognitions.append(image.shape))})
    FaceEmbeddingWorker._prime_inference(model)
    assert detections == [(settings.insightface_det_size, settings.insightface_det_size, 3)]
    assert recognitions == [(112,112,3)]
    assert meshes == [(settings.insightface_det_size, settings.insightface_det_size, 3), (270,240,3)]


def test_buffalo_loader_opens_only_the_existing_detection_and_recognition_weights(monkeypatch, tmp_path):
    from app.workers.face_embedding_worker import load_face_analysis

    class Analysis:
        def __init__(self, **kwargs):
            raise AssertionError('Do not initialize the unused model sessions')

    calls = []
    for name in ('det_10g.onnx', 'w600k_r50.onnx', 'unused_landmarks.onnx'):
        (tmp_path / name).touch()

    def load(path, **options):
        calls.append((path, options))
        return SimpleNamespace(taskname='detection' if path.endswith('det_10g.onnx') else 'recognition')

    monkeypatch.setitem(sys.modules, 'insightface.app', SimpleNamespace(FaceAnalysis=Analysis))
    monkeypatch.setitem(sys.modules, 'insightface', SimpleNamespace(model_zoo=SimpleNamespace(get_model=load)))
    monkeypatch.setitem(sys.modules, 'insightface.utils.storage', SimpleNamespace(ensure_available=lambda *args, **kwargs: str(tmp_path)))
    analysis = load_face_analysis(name='buffalo_l', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])
    assert set(analysis.models) == {'detection','recognition'}
    assert analysis.det_model is analysis.models['detection']
    assert [path.rsplit('\\',1)[-1].rsplit('/',1)[-1] for path, _ in calls] == ['det_10g.onnx','w600k_r50.onnx']
    assert all(options == {'providers':['CPUExecutionProvider']} for _, options in calls)


def test_health_remains_responsive_while_liveness_inference_is_running(monkeypatch):
    started, release = Event(), Event()

    def blocking_inference(payload):
        started.set()
        assert release.wait(5), "The event loop was blocked by inference"
        return FacialLivenessCheckResponse(live=False, reason="synthetic capture")

    app = create_app()
    app.dependency_overrides[facial_service] = lambda: SimpleNamespace(check_liveness=blocking_inference)
    monkeypatch.setattr(settings, "api_key", "runtime-test-internal-key")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = asyncio.create_task(client.post(
                "/api/v1/facial-embeddings/liveness-check",
                headers={"X-API-Key": settings.api_key},
                json={"image_frames": ["synthetic"] * 3, "liveness_challenge": "MOVE_UP"},
            ))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                health = await asyncio.wait_for(client.get("/health"), timeout=1)
                assert health.status_code == 200
            finally:
                release.set()
                response = await request
            assert response.status_code == 200
            assert response.json()["live"] is False

    asyncio.run(exercise())


def test_registration_runs_cpu_work_off_loop_and_keeps_persistence_on_loop(monkeypatch):
    main_thread = get_ident()
    generated = frames_for("MOVE_UP")[-1]
    service = FacialEmbeddingService(session=None, embedding_worker=SimpleNamespace(), repository=AsyncMock())
    persistence = AsyncMock(return_value="persisted")

    def inference(payload):
        assert get_ident() != main_thread
        return generated

    async def save(payload):
        assert get_ident() == main_thread
        return await persistence(payload)

    monkeypatch.setattr(service, "_generate_registration_embedding", inference)
    monkeypatch.setattr(service, "_register_embedding", save)
    payload = FacialEmbeddingFromImageCreate(user_id=uuid4(), image_base64="synthetic")

    assert asyncio.run(service.register_from_image(payload)) == "persisted"
    persistence.assert_awaited_once()


def test_warmup_and_all_request_threads_use_one_inference_owner(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    owner_calls = []
    caller = get_ident()
    worker = FaceEmbeddingWorker()
    monkeypatch.setattr(worker, '_get_model', lambda: object())
    monkeypatch.setattr(worker, '_prime_inference', lambda model: owner_calls.append(get_ident()))

    def infer(frame):
        owner_calls.append(get_ident())
        return frames_for('MOVE_UP')[0]

    monkeypatch.setattr(worker, '_generate', infer)
    try:
        worker.warm_up()
        with ThreadPoolExecutor(max_workers=4) as requests:
            results = list(requests.map(worker.generate, [f'owner-test:{index}' for index in range(8)]))
        assert len(results) == 8
        assert len(owner_calls) == 9
        assert len(set(owner_calls)) == 1
        assert owner_calls[0] != caller
    finally:
        worker.close()


def test_face_mesh_is_reused_only_on_its_own_thread(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local
    from app.workers import face_embedding_worker as module

    barrier = Barrier(2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, '_face_mesh_state', local())
    monkeypatch.setitem(sys.modules, 'mediapipe', SimpleNamespace(solutions=SimpleNamespace(
        face_mesh=SimpleNamespace(FaceMesh=lambda **options: object()))))

    def exercise(_):
        first = module.get_face_mesh()
        barrier.wait(timeout=5)
        assert module.get_face_mesh() is first
        return first

    with ThreadPoolExecutor(max_workers=2) as requests:
        results = list(requests.map(exercise, range(2)))
    assert results[0] is not results[1]
