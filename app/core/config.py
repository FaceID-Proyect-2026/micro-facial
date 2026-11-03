# Lee y centraliza la configuración del servicio.
from functools import lru_cache
import json
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FaceLit Embedding Service"
    environment: str = "production"
    api_key: str = ""
    database_url: str = Field(
        ...,
        description="SQLAlchemy async database URL.",
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
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
    inference_cache_size: int = Field(default=64, ge=0, le=256)
    inference_cache_ttl_seconds: float = Field(default=60.0, ge=0.0, le=120.0)
    preload_model: bool = False
    service_port: int = Field(
        default=8090,
        ge=1,
        le=65535,
        description="Puerto publicado por Compose; no cambia el puerto interno de Uvicorn.",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            value = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("CORS_ORIGINS debe ser una lista de orígenes HTTP/HTTPS explícitos.")
        origins = []
        for item in value:
            origin = item.strip().rstrip("/")
            if not origin:
                continue
            parsed = urlsplit(origin)
            if ("*" in origin or parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
                raise ValueError("Cada origen CORS debe indicar protocolo, servidor y puerto opcional, sin comodines ni rutas.")
            if origin not in origins:
                origins.append(origin)
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
