from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.schemas.facial_embedding import FacialVerificationRequest
from app.services.facial_embedding_service import FacialEmbeddingService
from app.workers.face_embedding_worker import GeneratedEmbedding
from tests.support.liveness_signals import frames_for


class FakeWorker:
    def __init__(self):
        self.calls = 0

    def generate(self, image_base64: str) -> GeneratedEmbedding:
        index = self.calls % 14
        self.calls += 1
        return frames_for("MOVE_UP" if index < 7 else "MOVE_DOWN")[index % 7]



class FakeRepository:
    def __init__(self, row):
        self.row = row
        self.candidate_apprentice_ids = None
        self.embedding = None

    async def find_best_session_match(self, *, candidate_apprentice_ids, embedding):
        self.candidate_apprentice_ids = candidate_apprentice_ids
        self.embedding = embedding
        return self.row


@pytest.mark.anyio
async def test_session_verification_returns_match_when_similarity_reaches_threshold():
    apprentice_id = uuid4()
    candidates = [apprentice_id, uuid4()]
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker(), repository=AsyncMock())
    service.repository = FakeRepository({"id_apprentice": apprentice_id, "similarity": 0.81})

    result = await service.verify_session_face(
        FacialVerificationRequest(
            candidate_apprentice_ids=candidates,
            image_frames=["data:image/jpeg;base64,abc"] * 14,
            liveness_challenges=["MOVE_UP", "MOVE_DOWN"],
            threshold=0.65,
        )
    )

    assert result.match is True
    assert result.id_apprentice == apprentice_id
    assert result.reason == "MATCH_FOUND"
    assert service.repository.candidate_apprentice_ids == candidates


@pytest.mark.anyio
async def test_session_verification_rejects_below_threshold():
    apprentice_id = uuid4()
    candidates = [apprentice_id]
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker(), repository=AsyncMock())
    service.repository = FakeRepository({"id_apprentice": apprentice_id, "similarity": 0.51})

    result = await service.verify_session_face(
        FacialVerificationRequest(
            candidate_apprentice_ids=candidates,
            image_frames=["data:image/jpeg;base64,abc"] * 14,
            liveness_challenges=["MOVE_UP", "MOVE_DOWN"],
            threshold=0.65,
        )
    )

    assert result.match is False
    assert result.id_apprentice == apprentice_id
    assert result.reason == "BELOW_THRESHOLD"


@pytest.mark.anyio
async def test_session_verification_rejects_without_active_candidates():
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker(), repository=AsyncMock())
    service.repository = FakeRepository(None)

    result = await service.verify_session_face(
        FacialVerificationRequest(
            image_frames=["data:image/jpeg;base64,abc"] * 14,
            liveness_challenges=["MOVE_UP", "MOVE_DOWN"],
            threshold=0.65,
        )
    )

    assert result.match is False
    assert result.id_apprentice is None
    assert result.reason == "NO_ACTIVE_CANDIDATES"


@pytest.mark.anyio
async def test_session_verification_rejects_single_photo_without_liveness_frames():
    service = FacialEmbeddingService(session=None, embedding_worker=FakeWorker(), repository=AsyncMock())
    service.repository = FakeRepository(None)

    result = await service.verify_session_face(
        FacialVerificationRequest(
            image_base64="data:image/jpeg;base64,abc",
            threshold=0.65,
        )
    )

    assert result.match is False
    assert result.live is False
    assert result.reason == "LIVENESS_REQUIRED"
