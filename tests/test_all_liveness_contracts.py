"""Four enabled challenge contracts, using measured-signal fixtures, not camera accuracy."""

import asyncio
from dataclasses import replace
from itertools import permutations
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.api.dependencies import facial_service
from app.core.config import settings
from app.main import create_app
from app.services.facial_embedding_service import FacialEmbeddingService
from app.services.liveness_validator import LivenessValidator


from tests.support.liveness_signals import CODES, MESH_CODES, SIGNALS, frames_for

PAIRS = list(permutations(CODES, 2))


@pytest.mark.parametrize("code", CODES)
def test_each_challenge_accepts_its_complete_seven_frame_signal(code):
    assert LivenessValidator().validate(frames_for(code), code)[0] is True


@pytest.mark.parametrize("expected,performed", PAIRS)
def test_no_other_challenge_can_substitute_for_the_requested_one(expected, performed):
    assert LivenessValidator().validate(frames_for(performed), expected)[0] is False


@pytest.mark.parametrize("code", CODES)
def test_each_challenge_rejects_a_still_face(code):
    assert LivenessValidator().validate(frames_for(), code)[0] is False


@pytest.mark.parametrize("first,second", PAIRS)
def test_every_distinct_pair_is_validated_in_capture_order(first, second):
    assert LivenessValidator().validate_sequence(
        frames_for(first) + frames_for(second), [first, second])[0] is True


@pytest.mark.parametrize("code", CODES)
def test_each_challenge_rejects_an_identity_change(code):
    frames = frames_for(code)
    frames[3] = replace(frames[3], embedding=[-.1] * 128)
    assert LivenessValidator().validate(frames, code)[0] is False


@pytest.mark.parametrize("code", MESH_CODES)
def test_mesh_challenges_reject_a_lost_mesh_frame(code):
    frames = frames_for(code)
    frames[3].image_metrics["mesh_available"] = 0.
    assert LivenessValidator().validate(frames, code)[0] is False


@pytest.mark.parametrize("code", MESH_CODES)
def test_missing_measurements_cannot_be_interpreted_as_a_gesture(code):
    key, _ = SIGNALS[code]
    frames = frames_for(code)
    del frames[3].image_metrics[key]
    assert LivenessValidator().validate(frames, code)[0] is False


@pytest.mark.parametrize("code", MESH_CODES)
@pytest.mark.parametrize("invalid", [None, float('nan'), float('inf'), 'invalid'])
def test_invalid_measurements_are_rejected_without_an_http_error(code, invalid):
    key, _ = SIGNALS[code]
    frames = frames_for(code)
    frames[3].image_metrics[key] = invalid
    assert LivenessValidator().validate(frames, code)[0] is False


@pytest.mark.parametrize("code", CODES)
def test_http_liveness_endpoint_uses_the_actual_challenge_rules(code, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "all-challenges-test-key")
    monkeypatch.setattr(settings, "preload_model", False)
    app = create_app()
    worker = MagicMock()
    worker.generate.side_effect = frames_for(code) + frames_for()
    service = FacialEmbeddingService(AsyncMock(), embedding_worker=worker)
    app.dependency_overrides[facial_service] = lambda: service

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = {"image_frames": ["signal-fixture"] * 7, "liveness_challenge": code}
            valid = await client.post("/api/v1/facial-embeddings/liveness-check", json=payload,
                                      headers={"X-API-Key": "all-challenges-test-key"})
            still = await client.post("/api/v1/facial-embeddings/liveness-check", json=payload,
                                      headers={"X-API-Key": "all-challenges-test-key"})
            assert valid.status_code == 200
            assert valid.json()["live"] is True
            assert still.status_code == 200
            assert still.json()["live"] is False
    asyncio.run(run())
