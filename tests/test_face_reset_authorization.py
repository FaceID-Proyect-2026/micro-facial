import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.facial_embedding_repository import FacialEmbeddingRepository
from app.schemas.facial_embedding import FacialEmbeddingCreate, FacialEmbeddingFromImageCreate, FacialEmbeddingPartialUpdate
from app.services.facial_embedding_service import FacialEmbeddingService


def result(value):
    response = MagicMock()
    response.scalar_one_or_none.return_value = value
    response.scalar_one.return_value = value
    return response


def test_first_registration_locks_owner_without_consuming_a_grant():
    session = AsyncMock()
    session.execute.side_effect = [result(uuid4()), result(False)]
    asyncio.run(FacialEmbeddingRepository(session).authorize_registration(uuid4(), False, None))
    assert session.execute.await_count == 2
    assert "FOR UPDATE OF u" in str(session.execute.call_args_list[0].args[0])


@pytest.mark.parametrize("replace,token", [(False, None), (True, None), (False, uuid4())])
def test_existing_face_requires_explicit_replacement_and_approval(replace, token):
    session = AsyncMock()
    session.execute.side_effect = [result(uuid4()), result(True)]
    with pytest.raises(ValueError, match="aprobación"):
        asyncio.run(FacialEmbeddingRepository(session).authorize_registration(uuid4(), replace, token))
    assert session.execute.await_count == 2


def test_grant_is_bound_to_owner_unexpired_and_consumed_atomically():
    session = AsyncMock()
    owner, token = uuid4(), uuid4()
    session.execute.side_effect = [result(owner), result(True), result(uuid4())]
    asyncio.run(FacialEmbeddingRepository(session).authorize_registration(uuid4(), True, token))
    query, params = session.execute.call_args.args
    assert params == {"user_id": owner, "token": token}
    sql = str(query)
    assert "request_status = 'ACCEPTED'" in sql
    assert "consumed_at IS NULL" in sql
    assert "authorization_expires_at >" in sql
    assert "SET consumed_at" in sql


def test_reused_expired_or_wrong_owner_token_cannot_be_consumed():
    session = AsyncMock()
    session.execute.side_effect = [result(uuid4()), result(True), result(None)]
    with pytest.raises(ValueError, match="expiró o ya fue utilizada"):
        asyncio.run(FacialEmbeddingRepository(session).authorize_registration(uuid4(), True, uuid4()))


def test_failed_embedding_write_rolls_back_authorization_consumption():
    session = AsyncMock()
    service = FacialEmbeddingService(session, embedding_worker=MagicMock())
    service.repository = AsyncMock()
    service.repository.resolve_apprentice_id.return_value = uuid4()
    service.repository.upsert.side_effect = RuntimeError("storage unavailable")
    payload = FacialEmbeddingCreate(user_id=uuid4(), embedding=[0.1] * 128, replace_existing=True, reset_token=uuid4())
    with pytest.raises(RuntimeError, match="storage unavailable"):
        asyncio.run(service._register_embedding(payload))
    service.repository.authorize_registration.assert_awaited_once()
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


def test_successful_replacement_writes_new_embedding_and_commits_once():
    session = AsyncMock()
    service = FacialEmbeddingService(session, embedding_worker=MagicMock())
    service.repository = AsyncMock()
    apprentice = uuid4()
    service.repository.resolve_apprentice_id.return_value = apprentice
    service.repository.upsert.return_value = {
        "id": uuid4(), "user_id": apprentice, "embedding": [0.2] * 128,
        "embedding_dimension": 128, "model_name": "test", "photo_reference": None,
        "status": "ACTIVE", "created_at": datetime.now(timezone.utc),
    }
    payload = FacialEmbeddingCreate(user_id=uuid4(), embedding=[0.2] * 128, replace_existing=True, reset_token=uuid4())
    asyncio.run(service._register_embedding(payload))
    service.repository.authorize_registration.assert_awaited_once_with(apprentice, True, payload.reset_token)
    assert service.repository.upsert.call_args.kwargs["embedding"] == [0.2] * 128
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


def test_approved_reregistration_still_requires_live_capture():
    service = FacialEmbeddingService(AsyncMock(), embedding_worker=MagicMock())
    payload = FacialEmbeddingFromImageCreate(user_id=uuid4(), image_base64="photo", reset_token=uuid4())
    with pytest.raises(ValueError, match="tres retos"):
        asyncio.run(service.register_from_image(payload))
    service.embedding_worker.generate.assert_not_called()


def test_raw_vector_and_partial_update_cannot_bypass_live_reregistration():
    service = FacialEmbeddingService(AsyncMock(), embedding_worker=MagicMock())
    raw = FacialEmbeddingCreate(user_id=uuid4(), embedding=[0.1] * 128, reset_token=uuid4())
    with pytest.raises(ValueError, match="captura en vivo"):
        asyncio.run(service.register(raw))
    with pytest.raises(ValueError, match="captura en vivo"):
        asyncio.run(service.update_partial(uuid4(), FacialEmbeddingPartialUpdate(embedding=[0.1] * 128)))
