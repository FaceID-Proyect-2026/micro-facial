# Representa el vector facial y las métricas derivadas de una imagen.
from dataclasses import dataclass


# Resultado de inferencia compartido por el trabajador y las reglas de aplicación.
@dataclass(frozen=True)
class GeneratedEmbedding:
    embedding: list[float]
    model_name: str
    face_count: int
    landmarks: list[list[float]]
    bbox: list[float]
    image_metrics: dict[str, float]
