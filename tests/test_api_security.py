import asyncio
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.api.dependencies import require_api_key
from app.core.config import settings
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
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


@pytest.mark.parametrize("environment,expected", [("local", 200), ("production", 404), ("ci", 404), ("staging", 404)])
def test_documentation_is_local_only(environment, expected, monkeypatch):
    monkeypatch.setattr(settings, "environment", environment)
    client = TestClient(create_app())
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == expected
    assert client.get("/health").status_code == 200


def test_cors_accepts_only_explicit_origin_without_browser_credentials(monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", ["https://facelit.example.invalid"])
    client = TestClient(create_app())
    headers = {"Origin": "https://facelit.example.invalid", "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "X-API-Key,Content-Type"}
    accepted = client.options("/api/v1/facial-embeddings/from-image", headers=headers)
    assert accepted.status_code == 200
    assert accepted.headers["access-control-allow-origin"] == headers["Origin"]
    assert "access-control-allow-credentials" not in accepted.headers
    for origin in ("https://untrusted.example.invalid", "http://192.168.1.4:8081"):
        rejected = client.options("/api/v1/facial-embeddings/from-image", headers={**headers, "Origin": origin})
        assert rejected.status_code == 400
        assert "access-control-allow-origin" not in rejected.headers


@pytest.mark.parametrize("value", ["*", "https://*.example.invalid", "https://example.invalid/path"])
def test_cors_settings_reject_wildcards_and_paths(value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, database_url="postgresql+asyncpg://test:test@localhost/test", cors_origins=value)


@pytest.mark.parametrize("value", ["https://a.example.invalid,http://localhost:8081", '["https://a.example.invalid", "http://localhost:8081"]'])
def test_cors_environment_accepts_csv_and_json(value, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", value)
    config = Settings(_env_file=None, database_url="postgresql+asyncpg://test:test@localhost/test")
    assert config.cors_origins == ["https://a.example.invalid", "http://localhost:8081"]
