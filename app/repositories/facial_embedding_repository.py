from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


class FacialEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def resolve_apprentice_id(self, user_id: UUID) -> UUID | None:
        result = await self.session.execute(
            text(
                """
                SELECT id_apprentice
                FROM academic.apprentice
                WHERE deleted_at IS NULL
                  AND (
                    id_apprentice = :user_id
                    OR id_user_app = :user_id
                  )
                LIMIT 1
                """
            ),
            {"user_id": user_id},
        )
        return result.scalar_one_or_none()

    async def get_active_by_user(self, user_id: UUID) -> dict[str, Any] | None:
        result = await self.session.execute(
            text(
                """
                SELECT
                    id_user_face AS id,
                    id_apprentice AS user_id,
                    embedding::REAL[] AS embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    created_at
                FROM facialrecognition.user_face
                WHERE id_apprentice = :user_id
                  AND status = 'ACTIVE'
                  AND deleted_at IS NULL
                """
            ),
            {"user_id": user_id},
        )
        row = result.mappings().one_or_none()
        return dict(row) if row else None

    async def upsert(
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
                    id_apprentice,
                    embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    created_by
                )
                VALUES (
                    :user_id,
                    CAST(:embedding AS vector),
                    :embedding_dimension,
                    :model_name,
                    :photo_reference,
                    'ACTIVE',
                    :created_by
                )
                ON CONFLICT (id_apprentice) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    embedding_dimension = EXCLUDED.embedding_dimension,
                    model_name = EXCLUDED.model_name,
                    photo_reference = EXCLUDED.photo_reference,
                    status = 'ACTIVE',
                    updated_at = NOW(),
                    updated_by = :created_by,
                    deleted_at = NULL,
                    deleted_by = NULL
                RETURNING
                    id_user_face AS id,
                    id_apprentice AS user_id,
                    embedding::REAL[] AS embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    created_at
                """
            ),
            {
                "user_id": user_id,
                "embedding": vector_literal(embedding),
                "embedding_dimension": len(embedding),
                "model_name": model_name,
                "photo_reference": photo_reference,
                "created_by": created_by,
            },
        )
        return dict(result.mappings().one())

    async def update_active(
        self,
        *,
        user_id: UUID,
        embedding: list[float] | None,
        model_name: str | None,
        photo_reference: str | None,
        update_photo_reference: bool,
        updated_by: str,
    ) -> dict[str, Any] | None:
        assignments = ["updated_at = NOW()", "updated_by = :updated_by"]
        params: dict[str, Any] = {"user_id": user_id, "updated_by": updated_by}

        if embedding is not None:
            assignments.extend(
                [
                    "embedding = CAST(:embedding AS vector)",
                    "embedding_dimension = :embedding_dimension",
                ]
            )
            params["embedding"] = vector_literal(embedding)
            params["embedding_dimension"] = len(embedding)

        if model_name is not None:
            assignments.append("model_name = :model_name")
            params["model_name"] = model_name

        if update_photo_reference:
            assignments.append("photo_reference = :photo_reference")
            params["photo_reference"] = photo_reference

        result = await self.session.execute(
            text(
                f"""
                UPDATE facialrecognition.user_face
                SET {", ".join(assignments)}
                WHERE id_apprentice = :user_id
                  AND status = 'ACTIVE'
                  AND deleted_at IS NULL
                RETURNING
                    id_user_face AS id,
                    id_apprentice AS user_id,
                    embedding::REAL[] AS embedding,
                    embedding_dimension,
                    model_name,
                    photo_reference,
                    status,
                    created_at
                """
            ),
            params,
        )
        row = result.mappings().one_or_none()
        return dict(row) if row else None

    async def list_active(self, limit: int, offset: int) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
                SELECT
                    id_user_face AS id,
                    id_apprentice AS user_id,
                    embedding_dimension,
                    model_name,
                    status,
                    created_at
                FROM facialrecognition.user_face
                WHERE status = 'ACTIVE'
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
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
                WHERE id_apprentice = :user_id
                  AND status = 'ACTIVE'
                  AND deleted_at IS NULL
                """
            ),
            {"user_id": user_id, "deleted_by": deleted_by},
        )
        return bool(result.rowcount)

    async def find_best_session_match(
        self,
        *,
        record_environment_id: UUID,
        embedding: list[float],
    ) -> dict[str, Any] | None:
        result = await self.session.execute(
            text(
                """
                SELECT
                    uf.id_apprentice,
                    1 - (uf.embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM facialrecognition.user_face uf
                JOIN academic.apprentice_chip ac
                  ON ac.id_apprentice = uf.id_apprentice
                JOIN environment.record_environment re
                  ON re.id_chip = ac.id_chip
                WHERE re.id_record_environment = :record_environment_id
                  AND re.active = TRUE
                  AND re.deleted_at IS NULL
                  AND ac.state = 'ACTIVE'
                  AND ac.deleted_at IS NULL
                  AND uf.status = 'ACTIVE'
                  AND uf.deleted_at IS NULL
                ORDER BY uf.embedding <=> CAST(:embedding AS vector)
                LIMIT 1
                """
            ),
            {
                "record_environment_id": record_environment_id,
                "embedding": vector_literal(embedding),
            },
        )
        row = result.mappings().one_or_none()
        return dict(row) if row else None
