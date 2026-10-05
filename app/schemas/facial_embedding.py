import math
from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings


class FacialEmbeddingCreate(BaseModel):
    user_id: UUID = Field(..., description="ID del aprendiz en security.user_app.")
    embedding: list[float] = Field(..., description="Vector facial generado por el modelo de reconocimiento.")
    model_name: str = Field(default="unknown", min_length=1, max_length=100)
    photo_reference: str | None = Field(default=None, max_length=500)
    replace_existing: bool = Field(
        default=False,
        description="Si es true, reemplaza el embedding activo anterior del aprendiz.",
    )
    created_by: str = Field(default="embedding-service", max_length=100)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        dimension = len(value)
        if dimension < settings.min_embedding_dimension:
            raise ValueError(f"El embedding debe tener al menos {settings.min_embedding_dimension} valores.")
        if dimension > settings.max_embedding_dimension:
            raise ValueError(f"El embedding no puede superar {settings.max_embedding_dimension} valores.")
        if not all(math.isfinite(item) for item in value):
            raise ValueError("El embedding contiene valores no finitos.")
        norm = math.sqrt(sum(item * item for item in value))
        if norm == 0:
            raise ValueError("El embedding no puede ser un vector cero.")
        return value


class FacialEmbeddingFromImageCreate(BaseModel):
    user_id: UUID = Field(..., description="ID del aprendiz en security.user_app.")
    image_base64: str = Field(..., min_length=1, description="Imagen facial en base64 o data URI.")
    photo_reference: str | None = Field(default=None, max_length=500)
    replace_existing: bool = Field(
        default=False,
        description="Si es true, reemplaza el embedding activo anterior del aprendiz.",
    )
    created_by: str = Field(default="embedding-service", max_length=100)


class FacialEmbeddingPartialUpdate(BaseModel):
    embedding: list[float] | None = Field(
        default=None,
        description="Nuevo vector facial. Si se envia, recalcula embedding_dimension.",
    )
    image_base64: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("image_base64", "imageBase64"),
        description="Imagen facial en base64 o data URI para regenerar el embedding.",
    )
    model_name: str | None = Field(default=None, min_length=1, max_length=100)
    photo_reference: str | None = Field(
        default=None,
        max_length=500,
        validation_alias=AliasChoices("photo_reference", "photoReference"),
    )
    updated_by: str = Field(
        default="embedding-service",
        max_length=100,
        validation_alias=AliasChoices("updated_by", "updatedBy"),
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        return FacialEmbeddingCreate.validate_embedding(value)


class FacialEmbeddingResponse(BaseModel):
    id: UUID
    user_id: UUID
    embedding: list[float]
    embedding_dimension: int
    model_name: str
    photo_reference: str | None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FacialEmbeddingSummary(BaseModel):
    id: UUID
    user_id: UUID
    embedding_dimension: int
    model_name: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
