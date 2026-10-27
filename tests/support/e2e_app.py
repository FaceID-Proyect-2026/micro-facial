"""HTTP integration fixture: real API, liveness rules, SQLAlchemy and PostgreSQL.

Only image inference is deterministic. Never use this test application in deployment.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import db_session, facial_service
from app.main import create_app
from app.services.facial_embedding_service import FacialEmbeddingService
from app.workers.face_embedding_worker import GeneratedEmbedding


class FixtureEmbeddingWorker:
    def __init__(self):
        self.calls = 0

    def generate(self, image_base64: str) -> GeneratedEmbedding:
        index = self.calls % 10
        self.calls += 1
        return GeneratedEmbedding(
            embedding=[0.1] * 128,
            model_name="integration-fixture",
            face_count=1,
            landmarks=[[30, 30], [70, 30]],
            bbox=[0, 0, 100, 100],
            image_metrics={
                "mesh_available": 1.0,
                "mesh_eye_ear": [.30, .28, .12, .28, .30][index] if index < 5 else .30,
                "mesh_mouth_open_ratio": [.02, .08, .25, .08, .02][index - 5] if index >= 5 else .02,
            },
        )


def fixture_service(session: AsyncSession = Depends(db_session)) -> FacialEmbeddingService:
    return FacialEmbeddingService(session, embedding_worker=FixtureEmbeddingWorker())


app = create_app()
app.dependency_overrides[facial_service] = fixture_service
