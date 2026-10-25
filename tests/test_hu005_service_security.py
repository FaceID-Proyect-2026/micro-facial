import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import db_session
from app.core.config import settings
from app.main import create_app
from app.schemas.facial_embedding import FacialEmbeddingResponse


@pytest.mark.parametrize("configured", ["", "   ", "server-test-secret"])
@pytest.mark.parametrize("method", ["get", "delete", "patch", "post"])
@pytest.mark.parametrize("supplied", [None, "incorrect"])
def test_routes_deny_access_before_database(configured, method, supplied, monkeypatch):
    monkeypatch.setattr(settings, "api_key", configured)
    app = create_app()

    async def forbidden_database():
        raise AssertionError("Unauthorized requests must never access the database")
        yield

    app.dependency_overrides[db_session] = forbidden_database
    client = TestClient(app)
    path = "/api/v1/facial-embeddings"
    if method != "post":
        path += f"/users/{uuid4()}"
    response = client.request(method, path, headers={} if supplied is None else {"X-API-Key": supplied})
    assert response.status_code == 401


@pytest.mark.parametrize("configured", ["", "  "])
def test_startup_requires_nonblank_key(configured, monkeypatch):
    monkeypatch.setattr(settings, "api_key", configured)
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(create_app()):
            pass


def test_valid_service_key_is_accepted(monkeypatch):
    from app.api.dependencies import require_api_key
    monkeypatch.setattr(settings, "api_key", "server-test-secret")
    assert asyncio.run(require_api_key("server-test-secret")) is None


def test_response_never_serializes_biometric_vector():
    record = dict(id=uuid4(), user_id=uuid4(), embedding=[0.1] * 128,
                  embedding_dimension=128, model_name="test", photo_reference=None,
                  status="ACTIVE", created_at=datetime.now(timezone.utc))
    response = FacialEmbeddingResponse.model_validate(record).model_dump(mode="json")
    assert "embedding" not in response
    assert response["embedding_dimension"] == 128


def test_authenticated_get_and_delete_keep_contract(monkeypatch):
    from unittest.mock import AsyncMock
    from app.api.v1.routes import facial_embeddings

    owner = uuid4()
    monkeypatch.setattr(settings, "api_key", "server-test-secret")
    service = AsyncMock()
    service.get_active_by_user.return_value = dict(
        id=uuid4(), user_id=owner, embedding=[0.1] * 128, embedding_dimension=128,
        model_name="test", photo_reference=None, status="ACTIVE", created_at=datetime.now(timezone.utc))
    service.deactivate.return_value = True
    monkeypatch.setattr(facial_embeddings, "FacialEmbeddingService", lambda session: service)
    app = create_app()

    async def fake_database():
        yield object()

    app.dependency_overrides[db_session] = fake_database
    with TestClient(app) as client:
        headers = {"X-API-Key": "server-test-secret"}
        path = f"/api/v1/facial-embeddings/users/{owner}"
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert "embedding" not in response.json()
        assert client.delete(path, headers=headers).status_code == 204
    service.deactivate.assert_awaited_once_with(owner, "embedding-service")
