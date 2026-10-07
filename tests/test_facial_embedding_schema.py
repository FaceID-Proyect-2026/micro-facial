from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.main import create_app
from app.schemas.facial_embedding import (
    FacialEmbeddingCreate,
    FacialEmbeddingPartialUpdate,
    FacialVerificationRequest,
)


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


def test_accepts_partial_update_payload():
    payload = FacialEmbeddingPartialUpdate(photo_reference=None)

    assert payload.photo_reference is None
    assert "photo_reference" in payload.model_fields_set


def test_accepts_partial_update_from_image_payload():
    payload = FacialEmbeddingPartialUpdate(
        image_base64="data:image/jpeg;base64,abc",
        photoReference="capture://updated.jpg",
        updatedBy="mobile-app",
    )

    assert payload.image_base64 == "data:image/jpeg;base64,abc"
    assert payload.photo_reference == "capture://updated.jpg"
    assert payload.updated_by == "mobile-app"


def test_accepts_session_verification_payload_aliases():
    record_environment_id = uuid4()
    payload = FacialVerificationRequest(
        recordEnvironmentId=record_environment_id,
        imageBase64="data:image/jpeg;base64,abc",
        threshold=0.72,
    )

    assert payload.record_environment_id == record_environment_id
    assert payload.image_base64 == "data:image/jpeg;base64,abc"
    assert payload.threshold == 0.72


def test_partial_update_rejects_invalid_embedding():
    with pytest.raises(ValidationError):
        FacialEmbeddingPartialUpdate(embedding=[0.0] * 128)


def test_openapi_includes_partial_update_endpoint():
    schema = create_app().openapi()

    patch_operation = schema["paths"]["/api/v1/facial-embeddings/users/{user_id}"]["patch"]
    verify_operation = schema["paths"]["/api/v1/facial-embeddings/verify-session"]["post"]
    response_properties = schema["components"]["schemas"]["FacialEmbeddingResponse"]["properties"]
    summary_properties = schema["components"]["schemas"]["FacialEmbeddingSummary"]["properties"]

    assert patch_operation["summary"] == "Actualizar parcialmente embedding facial activo por aprendiz"
    assert verify_operation["summary"] == "Verificar rostro temporal contra los aprendices de una sesion"
    assert (
        patch_operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/FacialEmbeddingPartialUpdate"
    )
    assert (
        verify_operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/FacialVerificationRequest"
    )
    partial_properties = schema["components"]["schemas"]["FacialEmbeddingPartialUpdate"]["properties"]
    assert "image_base64" in partial_properties
    assert "created_at" in response_properties
    assert "registered_at" not in response_properties
    assert "created_at" in summary_properties
    assert "registered_at" not in summary_properties
