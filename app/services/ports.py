"""Small structural contracts consumed by the application layer."""

from typing import Any, Protocol
from uuid import UUID

from app.domain.embedding import GeneratedEmbedding


class EmbeddingGenerator(Protocol):
    def generate(self, image_base64: str) -> GeneratedEmbedding: ...


class LivenessPolicy(Protocol):
    def validate_sequence(self, frames: list[GeneratedEmbedding], challenges: list[str]) -> tuple[bool, str]: ...

    def validate(self, frames: list[GeneratedEmbedding], challenge: str | None = None) -> tuple[bool, str]: ...


class FacialEmbeddingStore(Protocol):
    async def authorize_registration(self, apprentice_id: UUID, reset_owner_user_id: UUID | None,
                                     replace_existing: bool, reset_token: UUID | None) -> None: ...

    async def find_duplicate_embedding(self, *, apprentice_id: UUID, embedding: list[float], threshold: float) -> dict[str, Any] | None: ...

    async def get_active_by_user(self, user_id: UUID) -> dict[str, Any] | None: ...

    async def upsert(self, *, user_id: UUID, embedding: list[float], model_name: str,
                     photo_reference: str | None, created_by: str) -> dict[str, Any]: ...

    async def update_active(self, *, user_id: UUID, embedding: list[float] | None,
                            model_name: str | None, photo_reference: str | None,
                            update_photo_reference: bool, updated_by: str) -> dict[str, Any] | None: ...

    async def list_active(self, limit: int, offset: int) -> list[dict[str, Any]]: ...

    async def deactivate(self, user_id: UUID, deleted_by: str) -> bool: ...

    async def find_best_session_match(self, *, candidate_apprentice_ids: list[UUID], embedding: list[float]) -> dict[str, Any] | None: ...


class UnitOfWork(Protocol):
    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
