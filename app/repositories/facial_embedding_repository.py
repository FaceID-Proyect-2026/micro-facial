from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class FacialEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def user_exists(self, user_id: UUID) -> bool:
        result = await self.session.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM security.user_app
                    WHERE id_user_app = :user_id
                      AND deleted_at IS NULL
                )
                """
            ),
            {"user_id": user_id},
        )
        return bool(result.scalar_one())

    async def get_active_by_user(self, user_id: UUID) -> dict[str, Any] | None:
        result = await self.session.execute(
            text(
                """
                SELECT
                    id_user_face AS id,
                    id_user_app AS user_id,
                    embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    registered_at
                FROM facialrecognition.user_face
                WHERE id_user_app = :user_id
                  AND status = 'ACTIVE'
                  AND deleted_at IS NULL
                """
            ),
            {"user_id": user_id},
        )
        row = result.mappings().one_or_none()
        return dict(row) if row else None

    async def replace_active(self, user_id: UUID, updated_by: str) -> None:
        await self.session.execute(
            text(
                """
                UPDATE facialrecognition.user_face
                SET status = 'REPLACED',
                    updated_at = NOW(),
                    updated_by = :updated_by
                WHERE id_user_app = :user_id
                  AND status = 'ACTIVE'
                """
            ),
            {"user_id": user_id, "updated_by": updated_by},
        )

    async def create(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        model_name: str,
        photo_reference: str | None,
        created_by: str,
    ) -> dict[str, Any]:
        result = await self.session.execute(
            text(
                """
                INSERT INTO facialrecognition.user_face (
                    id_user_app,
                    embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    created_by
                )
                VALUES (
                    :user_id,
                    :embedding,
                    :embedding_dimension,
                    :model_name,
                    :photo_reference,
                    'ACTIVE',
                    :created_by
                )
                RETURNING
                    id_user_face AS id,
                    id_user_app AS user_id,
                    embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    registered_at
                """
            ),
            {
                "user_id": user_id,
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "model_name": model_name,
                "photo_reference": photo_reference,
                "created_by": created_by,
            },
        )
        return dict(result.mappings().one())

    async def list_active(self, limit: int, offset: int) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
                SELECT
                    id_user_face AS id,
                    id_user_app AS user_id,
                    embedding_dimension,
                    model_name,
                    status,
                    registered_at
                FROM facialrecognition.user_face
                WHERE status = 'ACTIVE'
                  AND deleted_at IS NULL
                ORDER BY registered_at DESC
                LIMIT :limit OFFSET :offset
                """
            ),
            {"limit": limit, "offset": offset},
        )
        return [dict(row) for row in result.mappings().all()]

    async def deactivate(self, user_id: UUID, deleted_by: str) -> bool:
        result = await self.session.execute(
            text(
                """
                UPDATE facialrecognition.user_face
                SET status = 'DELETED',
                    deleted_at = NOW(),
                    deleted_by = :deleted_by
                WHERE id_user_app = :user_id
                  AND status = 'ACTIVE'
                  AND deleted_at IS NULL
                """
            ),
            {"user_id": user_id, "deleted_by": deleted_by},
        )
        return bool(result.rowcount)
