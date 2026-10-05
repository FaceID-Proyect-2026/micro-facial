from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.facial_embedding import FacialEmbeddingCreate


def valid_payload(**overrides):
    payload = {
        "user_id": uuid4(),
        "embedding": [0.1] * 128,
        "model_name": "facenet-test",
    }
    payload.update(overrides)
    return payload


def test_accepts_valid_embedding_payload():
    payload = FacialEmbeddingCreate(**valid_payload())

    assert payload.user_id
    assert len(payload.embedding) == 128
    assert payload.model_name == "facenet-test"


@pytest.mark.parametrize(
    "embedding",
    [
        [],
        [0.0] * 128,
        [float("nan")] * 128,
        [float("inf")] * 128,
    ],
)
def test_rejects_invalid_embeddings(embedding):
    with pytest.raises(ValidationError):
        FacialEmbeddingCreate(**valid_payload(embedding=embedding))
