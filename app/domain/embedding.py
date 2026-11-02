from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedEmbedding:
    embedding: list[float]
    model_name: str
    face_count: int
    landmarks: list[list[float]]
    bbox: list[float]
    image_metrics: dict[str, float]
