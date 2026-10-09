from uuid import uuid4

import pytest

from app.schemas.facial_embedding import FacialVerificationRequest
from app.services.facial_embedding_service import FacialEmbeddingService
from app.workers.face_embedding_worker import GeneratedEmbedding


class FakeWorker:
    def __init__(self):
        self.calls = 0

    def generate(self, image_base64: str) -> GeneratedEmbedding:
        offset = self.calls * 3
        self.calls += 1
        return GeneratedEmbedding(
            embedding=[0.1] * 128,
            model_name="insightface/test",
            face_count=1,
            landmarks=[
                [30 + offset, 30],
                [70 - offset, 30],
                [50, 50 + offset],
                [35 + offset, 75],
                [65 - offset, 75],
            ],
            bbox=[0, 0, 100, 100],
            image_metrics={
                "eye_dark_ratio": 0.05 + (0.03 if self.calls == 2 else 0.0),
                "mouth_dark_ratio": 0.04 + (0.05 if self.calls == 3 else 0.0),
                "mouth_red_ratio": 0.01 + (0.02 if self.calls == 3 else 0.0),
            },
        )


class FakeRepository:
    def __init__(self, row):
        self.row = row
        self.record_environment_id = None
        self.embedding = None

    async def find_best_session_match(self, *, record_environment_id, embedding):
        self.record_environment_id = record_environment_id
        self.embedding = embedding
        return self.row


@pytest.mark.anyio
async def test_session_verification_returns_match_when_similarity_reaches_threshold():
    apprentice_id = uuid4()
    record_environment_id = uuid4()
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker())
    service.repository = FakeRepository({"id_apprentice": apprentice_id, "similarity": 0.81})

    result = await service.verify_session_face(
        FacialVerificationRequest(
            record_environment_id=record_environment_id,
            image_frames=["data:image/jpeg;base64,abc"] * 3,
            threshold=0.65,
        )
    )

    assert result.match is True
    assert result.id_apprentice == apprentice_id
    assert result.reason == "MATCH_FOUND"
    assert service.repository.record_environment_id == record_environment_id


@pytest.mark.anyio
async def test_session_verification_rejects_below_threshold():
    apprentice_id = uuid4()
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker())
    service.repository = FakeRepository({"id_apprentice": apprentice_id, "similarity": 0.51})

    result = await service.verify_session_face(
        FacialVerificationRequest(
            record_environment_id=uuid4(),
            image_frames=["data:image/jpeg;base64,abc"] * 3,
            threshold=0.65,
        )
    )

    assert result.match is False
    assert result.id_apprentice == apprentice_id
    assert result.reason == "BELOW_THRESHOLD"


@pytest.mark.anyio
async def test_session_verification_rejects_without_active_candidates():
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker())
    service.repository = FakeRepository(None)

    result = await service.verify_session_face(
        FacialVerificationRequest(
            record_environment_id=uuid4(),
            image_frames=["data:image/jpeg;base64,abc"] * 3,
            threshold=0.65,
        )
    )

    assert result.match is False
    assert result.id_apprentice is None
    assert result.reason == "NO_ACTIVE_CANDIDATES"


@pytest.mark.anyio
async def test_session_verification_rejects_single_photo_without_liveness_frames():
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker())
    service.repository = FakeRepository(None)

    result = await service.verify_session_face(
        FacialVerificationRequest(
            record_environment_id=uuid4(),
            image_base64="data:image/jpeg;base64,abc",
            threshold=0.65,
        )
    )

    assert result.match is False
    assert result.live is False
    assert result.reason == "LIVENESS_REQUIRED"
