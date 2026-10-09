from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FaceLit Embedding Service"
    environment: str = "local"
    api_key: str = ""
    database_url: str = Field(
        default="postgresql+asyncpg://facelit_user:facelit_password@localhost:5439/facelit-db",
        description="SQLAlchemy async database URL.",
    )
    cors_origins: list[str] = ["http://localhost:8081", "http://localhost:19006"]
    cors_origin_regex: str = (
        r"https?://("
        r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
        r"10(?:\.\d{1,3}){3}|"
        r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}|"
        r"192\.168(?:\.\d{1,3}){2}"
        r"):\d+"
    )
    min_embedding_dimension: int = 32
    max_embedding_dimension: int = 4096
    facial_match_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Similaridad coseno minima para aceptar una coincidencia facial.",
    )
    liveness_min_frames: int = Field(
        default=3,
        ge=2,
        le=8,
        description="Cantidad minima de frames requeridos para validar vida en asistencia.",
    )
    liveness_max_frames: int = Field(
        default=15,
        ge=2,
        le=24,
        description="Cantidad maxima de frames que se procesan para validar vida.",
    )
    liveness_min_landmark_motion: float = Field(
        default=0.003,
        ge=0.0,
        le=0.25,
        description="Movimiento minimo normalizado de landmarks para rechazar fotos estaticas.",
    )
    liveness_min_box_motion: float = Field(
        default=0.008,
        ge=0.0,
        le=0.5,
        description="Movimiento minimo normalizado del encuadre del rostro para aceptar vida.",
    )
    liveness_min_embedding_similarity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Similaridad minima entre frames para confirmar que la secuencia es de la misma persona.",
    )
    insightface_model_name: str = "buffalo_l"
    insightface_det_size: int = 640
    insightface_ctx_id: int = -1
    preload_model: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
