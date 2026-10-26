import asyncio
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.api.dependencies import require_api_key
from app.core.config import settings
from app.main import create_app
from app.schemas.facial_embedding import FacialEmbeddingResponse


@pytest.mark.parametrize("configured,supplied", [("", None), ("", "anything"), ("change-me", "change-me"), ("server-only-test-key", None), ("server-only-test-key", "wrong"), ("server-only-test-key", "ñ")])
def test_guard_never_becomes_public(configured, supplied, monkeypatch):
    monkeypatch.setattr(settings, "api_key", configured)
    with pytest.raises(HTTPException) as error:
        asyncio.run(require_api_key(supplied))
    assert error.value.status_code == 401


def test_matching_service_credential_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "server-only-test-key")
    asyncio.run(require_api_key("server-only-test-key"))


@pytest.mark.parametrize("key", ["", "   ", "change-me"])
def test_startup_refuses_unconfigured_service_key(key, monkeypatch):
    monkeypatch.setattr(settings, "api_key", key)
    with pytest.raises(RuntimeError, match="API_KEY"):
        asyncio.run(create_app().router.startup())


def test_embedding_response_does_not_disclose_vector():
    response = FacialEmbeddingResponse.model_validate(dict(id=uuid4(), user_id=uuid4(), embedding=[0.1]*128,
        embedding_dimension=128, model_name="synthetic", photo_reference=None, status="ACTIVE", created_at="2026-10-26T10:00:00Z"))
    assert "embedding" not in response.model_dump()


def test_validation_errors_are_serializable_and_do_not_echo_biometrics():
    app = create_app()
    error = RequestValidationError([{"type": "value_error", "loc": ("body", "image_frames"), "msg": "Invalid capture",
        "input": "PRIVATE_BIOMETRIC_INPUT", "ctx": {"error": ValueError("private")}}])
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/facial-embeddings/from-image", "headers": []})
    response = asyncio.run(app.exception_handlers[RequestValidationError](request, error))
    assert response.status_code == 422
    assert "PRIVATE_BIOMETRIC_INPUT" not in response.body.decode()
    assert json.loads(response.body)["detail"][0]["loc"] == ["body", "image_frames"]
