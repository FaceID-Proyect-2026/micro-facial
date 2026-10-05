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
    min_embedding_dimension: int = 32
    max_embedding_dimension: int = 4096
    insightface_model_name: str = "buffalo_l"
    insightface_det_size: int = 640
    insightface_ctx_id: int = -1

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
