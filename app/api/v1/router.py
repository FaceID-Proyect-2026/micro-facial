from fastapi import APIRouter

from app.api.v1.routes import facial_embeddings

api_router = APIRouter()
api_router.include_router(facial_embeddings.router, prefix="/facial-embeddings", tags=["facial embeddings"])
