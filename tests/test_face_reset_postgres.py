"""Opt-in integration test. All fixtures and biometric vectors are rolled back.

RF54_POSTGRES_TEST=1 python -m pytest tests/test_face_reset_postgres.py -q
Uses the local DATABASE_URL; refuses remote database hosts.
"""
import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.repositories.facial_embedding_repository import FacialEmbeddingRepository


@pytest.mark.skipif(os.getenv("RF54_POSTGRES_TEST") != "1", reason="Requires opt-in local PostgreSQL")
def test_real_postgres_replacement_consumption_and_rollback():
    asyncio.run(exercise())


async def exercise():
    url = make_url(settings.database_url)
    assert url.host in {"localhost", "127.0.0.1"}, "Integration test requires a local PostgreSQL host"
    engine = create_async_engine(url)
    schema = "rf54_test_" + uuid4().hex
    try:
        async with AsyncSession(engine) as session:
            # One outer transaction: rollback removes the schema, fixtures and vectors.
            await session.execute(text(f'CREATE SCHEMA "{schema}"'))
            await session.execute(text(f'CREATE TABLE "{schema}".user_app (id_user_app uuid PRIMARY KEY)'))
            await session.execute(text(f'CREATE TABLE "{schema}".apprentice (id_apprentice uuid PRIMARY KEY, id_user_app uuid, deleted_at timestamptz)'))
            await session.execute(text(f'CREATE TABLE "{schema}".user_face (LIKE facialrecognition.user_face INCLUDING ALL)'))
            await session.execute(text(f'CREATE TABLE "{schema}".face_reset_request (LIKE facialrecognition.face_reset_request INCLUDING ALL)'))

            class IsolatedSession:
                async def execute(self, statement, params=None):
                    sql = str(statement)
                    for source, table in [("security.user_app", "user_app"), ("academic.apprentice", "apprentice"),
                                          ("facialrecognition.user_face", "user_face"), ("facialrecognition.face_reset_request", "face_reset_request")]:
                        sql = sql.replace(source, f'"{schema}".{table}')
                    return await session.execute(text(sql), params or {})

            repo = FacialEmbeddingRepository(IsolatedSession())
            owner, apprentice, request_id, token = uuid4(), uuid4(), uuid4(), uuid4()
            await session.execute(text(f'INSERT INTO "{schema}".user_app VALUES (:id)'), {"id": owner})
            await session.execute(text(f'INSERT INTO "{schema}".apprentice VALUES (:id, :owner, NULL)'), {"id": apprentice, "owner": owner})
            await repo.authorize_registration(apprentice, False, None)
            original = await repo.upsert(user_id=apprentice, embedding=[0.1] * 128, model_name="test", photo_reference=None, created_by=str(owner))
            with pytest.raises(ValueError, match="aprobación"):
                await repo.authorize_registration(apprentice, True, None)
            await session.execute(text(f"""INSERT INTO "{schema}".face_reset_request
                (id_face_reset_request, id_user_app, request_status, resolved_by, resolved_at, authorization_token, authorization_expires_at)
                VALUES (:id, :owner, 'ACCEPTED', :owner, NOW(), :token, NOW() + INTERVAL '24 hours')"""),
                {"id": request_id, "owner": owner, "token": token})
            # A failed vector save must leave the authorization reusable.
            with pytest.raises(Exception, match="different vector dimensions|finite|NaN|infinite"):
                async with session.begin_nested():
                    await repo.authorize_registration(apprentice, True, token)
                    await repo.upsert(user_id=apprentice, embedding=[float("inf")] * 128, model_name="test", photo_reference=None, created_by=str(owner))
            consumed = await session.scalar(text(f'SELECT consumed_at FROM "{schema}".face_reset_request WHERE id_face_reset_request = :id'), {"id": request_id})
            assert consumed is None
            await repo.authorize_registration(apprentice, True, token)
            replacement = await repo.upsert(user_id=apprentice, embedding=[0.2] * 128, model_name="test-new", photo_reference=None, created_by=str(owner))
            assert replacement["id"] == original["id"]
            assert replacement["model_name"] == "test-new"
            assert replacement["embedding"][0] == pytest.approx(0.2)
            assert await session.scalar(text(f'SELECT COUNT(*) FROM "{schema}".user_face')) == 1
            with pytest.raises(ValueError, match="ya fue utilizada"):
                await repo.authorize_registration(apprentice, True, token)
            # Reset fixture only: test expiration and a token belonging to another owner.
            await session.execute(text(f'UPDATE "{schema}".face_reset_request SET consumed_at = NULL, authorization_expires_at = NOW() - INTERVAL \'1 second\''))
            with pytest.raises(ValueError, match="expiró"):
                await repo.authorize_registration(apprentice, True, token)
            with pytest.raises(ValueError, match="no existe"):
                await repo.authorize_registration(apprentice, True, uuid4())
            await session.rollback()
    finally:
        await engine.dispose()
