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
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=Model))
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
    service = FacialEmbeddingService(session=None, embedding_worker=SimpleNamespace())
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
