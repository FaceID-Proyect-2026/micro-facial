from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.schemas.facial_embedding import FacialEmbeddingFromImageCreate
from app.services.facial_embedding_service import FacialEmbeddingService
from app.workers.face_embedding_worker import GeneratedEmbedding


def sequence(key=None, values=None, mesh=True):
    baseline = {
        'mesh_available': float(mesh), 'mesh_eye_ear': 0.30, 'mesh_mouth_open_ratio': 0.02,
        'mesh_mouth_width_ratio': 0.4, 'mesh_mouth_white_ratio': 0.01,
        'mesh_nose_x': 0.0, 'mesh_nose_y': 0.5,
    }
    frames = []
    for value in values or [0] * 5:
        metrics = dict(baseline)
        if key:
            metrics[key] = value
        frames.append(GeneratedEmbedding([0.1] * 128, 'test', 1, [[30, 30], [70, 30]], [0, 0, 100, 100], metrics))
    return frames


def service():
    return FacialEmbeddingService(AsyncMock(), embedding_worker=MagicMock())


@pytest.mark.parametrize('code,key,values', [
    ('BLINK', 'mesh_eye_ear', [.3, .28, .12, .28, .3]),
    ('OPEN_CLOSE_MOUTH', 'mesh_mouth_open_ratio', [.02, .08, .25, .08, .02]),
    ('NOD', 'mesh_nose_y', [.5, .53, .6, .53, .5]),
    ('SHAKE_HEAD', 'mesh_nose_x', [0, .05, .1, .03, 0]),
    ('KISS_CAMERA', 'mesh_mouth_width_ratio', [.4, .37, .31, .37, .4]),
    ('MOVE_UP', 'mesh_nose_y', [.5, .48, .45, .43, .42]),
    ('MOVE_DOWN', 'mesh_nose_y', [.5, .52, .55, .57, .58]),
])
def test_requested_expression_requires_its_own_temporal_signal(code, key, values):
    assert service()._validate_liveness(sequence(key, values), code)[0] is True


@pytest.mark.parametrize('code', ['BLINK', 'OPEN_CLOSE_MOUTH', 'NOD', 'SHAKE_HEAD', 'KISS_CAMERA',
                                 'MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_CLOSER', 'MOVE_AWAY'])
def test_still_face_cannot_pass_any_named_challenge(code):
    assert service()._validate_liveness(sequence(), code)[0] is False


@pytest.mark.parametrize('code', [None, '', 'ANY_MOVEMENT', 'STICK_OUT_TONGUE', 'TYPO'])
def test_unspecified_or_unsupported_challenge_never_passes(code):
    assert service()._validate_liveness(sequence(), code)[0] is False


def test_smiling_is_not_a_blink_or_a_kiss():
    frames = sequence('mesh_mouth_width_ratio', [.4, .46, .5, .46, .4])
    assert service()._validate_liveness(frames, 'BLINK')[0] is False
    assert service()._validate_liveness(frames, 'KISS_CAMERA')[0] is False


def test_light_variation_without_mesh_cannot_substitute_for_expression():
    for code in ['BLINK', 'OPEN_CLOSE_MOUTH', 'MOVE_UP']:
        frames = sequence('eye_dark_ratio', [.01, .1, .2, .1, .01], mesh=False)
        assert service()._validate_liveness(frames, code)[0] is False


@pytest.mark.parametrize('code,key,values', [
    ('BLINK', 'mesh_eye_ear', [.3, .28, .12, .12, .12]),
    ('OPEN_CLOSE_MOUTH', 'mesh_mouth_open_ratio', [.02, .08, .25, .25, .25]),
    ('NOD', 'mesh_nose_y', [.5, .53, .6, .6, .6]),
    ('SHAKE_HEAD', 'mesh_nose_x', [0, .05, .1, .1, .1]),
])
def test_cycle_actions_must_finish_in_neutral_position(code, key, values):
    assert service()._validate_liveness(sequence(key, values), code)[0] is False


def test_nodding_does_not_count_as_holding_an_upward_rotation():
    frames = sequence('mesh_nose_y', [.5, .45, .4, .45, .5])
    assert service()._validate_liveness(frames, 'MOVE_UP')[0] is False


@pytest.mark.parametrize('codes', [[], ['BLINK'], ['BLINK', 'BLINK'], ['BLINK', 'NOD', 'KISS_CAMERA']])
def test_final_validation_requires_exactly_two_distinct_challenges(codes):
    assert service()._validate_liveness_sequence(sequence() * 2, codes)[0] is False


def test_final_sequence_revalidates_both_successes_and_identity():
    frames = sequence('mesh_eye_ear', [.3, .28, .12, .28, .3]) + sequence('mesh_mouth_open_ratio', [.02, .08, .25, .08, .02])
    assert service()._validate_liveness_sequence(frames, ['BLINK', 'OPEN_CLOSE_MOUTH'])[0] is True
    assert service()._validate_liveness_sequence(sequence() + frames[5:], ['BLINK', 'OPEN_CLOSE_MOUTH'])[0] is False
    frames[-1] = replace(frames[-1], embedding=[-0.1] * 128)
    assert service()._validate_liveness_sequence(frames, ['BLINK', 'OPEN_CLOSE_MOUTH'])[0] is False


def test_registration_cannot_bypass_two_challenges_with_a_single_photo():
    payload = FacialEmbeddingFromImageCreate(user_id=uuid4(), image_base64='photo')
    with pytest.raises(ValueError, match='dos retos'):
        service()._generate_registration_embedding(payload)


@pytest.mark.parametrize('code,box', [
    ('MOVE_LEFT', [-10, 0, 90, 100]), ('MOVE_RIGHT', [10, 0, 110, 100]),
    ('MOVE_CLOSER', [-5, -5, 105, 105]), ('MOVE_AWAY', [5, 5, 95, 95]),
])
def test_directional_challenges_accept_the_requested_direction(code, box):
    frames = sequence()
    frames[-1] = replace(frames[-1], bbox=box)
    assert service()._validate_liveness(frames, code)[0] is True


def test_registration_with_two_valid_challenges_uses_existing_enrollment_path():
    frames = sequence('mesh_eye_ear', [.3, .28, .12, .28, .3]) + sequence('mesh_mouth_open_ratio', [.02, .08, .25, .08, .02])
    instance = service()
    instance.embedding_worker.generate.side_effect = frames
    payload = FacialEmbeddingFromImageCreate(user_id=uuid4(), image_base64='photo',
                                            image_frames=['frame'] * 10,
                                            liveness_challenges=['BLINK', 'OPEN_CLOSE_MOUTH'])
    assert instance._generate_registration_embedding(payload) is frames[-1]


def test_completely_closed_eyes_still_count_as_a_blink_when_they_reopen():
    assert service()._validate_liveness(sequence('mesh_eye_ear', [.3, .28, 0, .28, .3]), 'BLINK')[0] is True
