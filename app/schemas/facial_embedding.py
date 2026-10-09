import math
from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings


class FacialEmbeddingCreate(BaseModel):
    user_id: UUID = Field(..., description="ID del aprendiz en academic.apprentice o id_user_app del usuario.")
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
    user_id: UUID = Field(..., description="ID del aprendiz en academic.apprentice o id_user_app del usuario.")
    image_base64: str = Field(..., min_length=1, description="Imagen facial en base64 o data URI.")
    image_frames: list[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=18,
        validation_alias=AliasChoices("image_frames", "imageFrames"),
        description="Secuencia opcional de frames capturados en vivo antes de guardar el embedding.",
    )
    liveness_challenge: str | None = Field(
        default=None,
        max_length=40,
        validation_alias=AliasChoices("liveness_challenge", "livenessChallenge"),
        description="Reto solicitado por la app para validar vida antes de guardar el embedding.",
    )
    liveness_challenges: list[str] = Field(
        default_factory=list,
        max_length=3,
        validation_alias=AliasChoices("liveness_challenges", "livenessChallenges"),
        description="Secuencia de retos solicitados por la app para validar vida paso a paso.",
    )
    photo_reference: str | None = Field(default=None, max_length=500)
    replace_existing: bool = Field(
        default=False,
        description="Si es true, reemplaza el embedding activo anterior del aprendiz.",
    )
    created_by: str = Field(default="embedding-service", max_length=100)

    model_config = ConfigDict(populate_by_name=True)


class FacialVerificationRequest(BaseModel):
    record_environment_id: UUID = Field(
        ...,
        validation_alias=AliasChoices("record_environment_id", "recordEnvironmentId"),
        description="Sesion activa creada por el instructor en environment.record_environment.",
    )
    image_base64: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("image_base64", "imageBase64"),
        description="Imagen capturada durante la sesion. Se mantiene por compatibilidad; asistencia debe enviar image_frames.",
    )
    image_frames: list[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=18,
        validation_alias=AliasChoices("image_frames", "imageFrames"),
        description="Secuencia de frames capturados en vivo para validar vida.",
    )
    liveness_challenge: str | None = Field(
        default=None,
        max_length=40,
        validation_alias=AliasChoices("liveness_challenge", "livenessChallenge"),
        description="Reto solicitado por la app para validar movimiento vivo.",
    )
    liveness_challenges: list[str] = Field(
        default_factory=list,
        max_length=3,
        validation_alias=AliasChoices("liveness_challenges", "livenessChallenges"),
        description="Secuencia de retos solicitados por la app para validar movimiento vivo paso a paso.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Umbral opcional de similaridad coseno. Si no se envia usa la configuracion del servicio.",
    )

    model_config = ConfigDict(populate_by_name=True)


class FacialLivenessCheckRequest(BaseModel):
    image_frames: list[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=18,
        validation_alias=AliasChoices("image_frames", "imageFrames"),
        description="Frames capturados para validar un reto de vida puntual.",
    )
    liveness_challenge: str | None = Field(
        default=None,
        max_length=40,
        validation_alias=AliasChoices("liveness_challenge", "livenessChallenge"),
        description="Reto solicitado por la app para validar el bloque actual.",
    )

    model_config = ConfigDict(populate_by_name=True)


class FacialLivenessCheckResponse(BaseModel):
    live: bool
    reason: str


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


class FacialVerificationResponse(BaseModel):
    match: bool
    live: bool = False
    id_apprentice: UUID | None = None
    similarity: float | None = None
    threshold: float
    model_name: str
    reason: str
    liveness_reason: str | None = None
