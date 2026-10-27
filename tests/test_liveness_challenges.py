from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.schemas.facial_embedding import FacialEmbeddingFromImageCreate
from app.services.facial_embedding_service import FacialEmbeddingService
from tests.support.liveness_signals import CODES, frames_for


REMOVED_CODES = ['BLINK', 'OPEN_CLOSE_MOUTH', 'NOD', 'SHAKE_HEAD', 'KISS_CAMERA', 'MOVE_LEFT', 'MOVE_RIGHT']


def service():
    return FacialEmbeddingService(AsyncMock(), embedding_worker=MagicMock())


@pytest.mark.parametrize('code', CODES)
def test_directional_challenges_accept_only_the_requested_movement(code):
    assert service()._validate_liveness(frames_for(code), code)[0] is True
    assert service()._validate_liveness(frames_for(), code)[0] is False


@pytest.mark.parametrize('code', [None, '', 'ANY_MOVEMENT', 'STICK_OUT_TONGUE', 'TYPO', *REMOVED_CODES])
def test_unspecified_removed_or_unsupported_challenge_never_passes(code):
    assert service()._validate_liveness(frames_for('MOVE_UP'), code)[0] is False


def test_light_variation_without_mesh_cannot_substitute_for_an_upward_rotation():
    frames = frames_for('MOVE_UP')
    for frame in frames:
        frame.image_metrics['mesh_available'] = 0.
    assert service()._validate_liveness(frames, 'MOVE_UP')[0] is False


def test_nodding_does_not_count_as_holding_an_upward_rotation():
    frames = frames_for('MOVE_UP')
    frames[-1].image_metrics['mesh_nose_y'] = .50
    assert service()._validate_liveness(frames, 'MOVE_UP')[0] is False


@pytest.mark.parametrize('codes', [[], ['MOVE_UP'], ['MOVE_UP', 'MOVE_UP'], ['MOVE_UP', 'MOVE_DOWN', 'MOVE_AWAY']])
def test_final_validation_requires_exactly_two_distinct_challenges(codes):
    assert service()._validate_liveness_sequence(frames_for() * 2, codes)[0] is False


@pytest.mark.parametrize('removed', REMOVED_CODES)
def test_removed_challenges_cannot_be_used_in_the_final_registration(removed):
    frames = frames_for('MOVE_UP') + frames_for('MOVE_DOWN')
    assert service()._validate_liveness_sequence(frames, ['MOVE_UP', removed])[0] is False


def test_final_sequence_revalidates_both_successes_and_identity():
    frames = frames_for('MOVE_UP') + frames_for('MOVE_DOWN')
    assert service()._validate_liveness_sequence(frames, ['MOVE_UP', 'MOVE_DOWN'])[0] is True
    assert service()._validate_liveness_sequence(frames_for() + frames[7:], ['MOVE_UP', 'MOVE_DOWN'])[0] is False
    frames[-1] = replace(frames[-1], embedding=[-0.1] * 128)
    assert service()._validate_liveness_sequence(frames, ['MOVE_UP', 'MOVE_DOWN'])[0] is False


def test_registration_cannot_bypass_two_challenges_with_a_single_photo():
    payload = FacialEmbeddingFromImageCreate(user_id=uuid4(), image_base64='photo')
    with pytest.raises(ValueError, match='dos retos'):
        service()._generate_registration_embedding(payload)


def test_registration_with_two_valid_challenges_uses_existing_enrollment_path():
    frames = frames_for('MOVE_UP') + frames_for('MOVE_DOWN')
    instance = service()
    instance.embedding_worker.generate.side_effect = frames
    payload = FacialEmbeddingFromImageCreate(user_id=uuid4(), image_base64='photo',
                                            image_frames=['frame'] * 14,
                                            liveness_challenges=['MOVE_UP', 'MOVE_DOWN'])
    assert instance._generate_registration_embedding(payload) is frames[-1]
