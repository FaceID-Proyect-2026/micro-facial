"""HTTP integration fixture: real API, liveness rules, SQLAlchemy and PostgreSQL.

Only image inference is deterministic. Never use this test application in deployment.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import db_session, facial_service
from app.main import create_app
from app.services.facial_embedding_service import FacialEmbeddingService
from app.workers.face_embedding_worker import GeneratedEmbedding
from liveness_signals import CODES, frames_for


class FixtureEmbeddingWorker:
    def __init__(self):
        self.calls = 0

    def generate(self, image_base64: str) -> GeneratedEmbedding:
        if image_base64.startswith("liveness-fixture:"):
            _, code, index = image_base64.split(":")
            if code != "STILL" and code not in CODES:
                raise ValueError("Unsupported integration-fixture challenge")
            return frames_for(code)[int(index)]
        index = self.calls % 14
        self.calls += 1
        code = "MOVE_UP" if index < 7 else "MOVE_DOWN"
        return frames_for(code)[index % 7]



def fixture_service(session: AsyncSession = Depends(db_session)) -> FacialEmbeddingService:
    return FacialEmbeddingService(session, embedding_worker=FixtureEmbeddingWorker())


app = create_app()
app.dependency_overrides[facial_service] = fixture_service
