"""Reuse inference only; authorization, identity matching and gestures remain live rules."""

import pytest
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.services.liveness_validator import LivenessValidator
from app.workers import face_embedding_worker as module
from tests.support.liveness_signals import frames_for


def configured_worker(monkeypatch, *, size=64, ttl=60):
    monkeypatch.setattr(settings, 'inference_cache_size', size)
    monkeypatch.setattr(settings, 'inference_cache_ttl_seconds', ttl)
    worker = module.FaceEmbeddingWorker()
    calls = []

    def generate(frame):
        calls.append(frame)
        return frames_for('MOVE_UP')[int(frame.split(':')[-1]) % 7]

    monkeypatch.setattr(worker, '_generate', generate)
    return worker, calls


def test_second_validation_reuses_exact_frames_without_accepting_wrong_gesture(monkeypatch):
    worker, calls = configured_worker(monkeypatch)
    captured = [f'capture:{index}' for index in range(7)]
    first = [worker.generate(frame) for frame in captured]
    second = [worker.generate(frame) for frame in captured]
    assert len(calls) == 7
    assert first == second
    assert LivenessValidator().validate(second, 'MOVE_UP')[0] is True
    assert LivenessValidator().validate(second, 'MOVE_DOWN')[0] is False
    worker.generate('different-capture:0')
    assert len(calls) == 8


def test_expired_features_are_recomputed_and_hits_do_not_extend_expiry(monkeypatch):
    worker, calls = configured_worker(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    worker.generate('frame:0')
    clock[0] = 59
    worker.generate('frame:0')
    clock[0] = 60
    worker.generate('frame:0')
    assert len(calls) == 2


def test_memory_bound_evicts_least_recent_frame(monkeypatch):
    worker, calls = configured_worker(monkeypatch, size=2)
    for frame in ('frame:0', 'frame:1', 'frame:0', 'frame:2', 'frame:1'):
        worker.generate(frame)
    assert calls == ['frame:0', 'frame:1', 'frame:2', 'frame:1']
    assert len(worker._recent_frames) == 2
    assert all(isinstance(key, bytes) and len(key) == 32 for key in worker._recent_frames)


@pytest.mark.parametrize('size,ttl', [(0, 60), (64, 0)])
def test_cache_can_be_disabled(monkeypatch, size, ttl):
    worker, calls = configured_worker(monkeypatch, size=size, ttl=ttl)
    worker.generate('frame:0')
    worker.generate('frame:0')
    assert len(calls) == 2
    assert not worker._recent_frames


def test_failed_inference_is_never_cached(monkeypatch):
    worker, _ = configured_worker(monkeypatch)

    def fail(frame):
        raise ValueError('more than one face')

    monkeypatch.setattr(worker, '_generate', fail)
    with pytest.raises(ValueError):
        worker.generate('frame:0')
    assert not worker._recent_frames


def test_simultaneous_rechecks_compute_the_same_frame_only_once(monkeypatch):
    worker, calls = configured_worker(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker.generate, ['frame:0'] * 8))
    assert len(calls) == 1
    assert all(result == results[0] for result in results)
