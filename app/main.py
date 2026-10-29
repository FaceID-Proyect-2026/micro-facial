import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.api.v1.router import api_router
from app.core.config import settings
from app.workers.face_embedding_worker import get_face_embedding_worker

logger = logging.getLogger("facelit.embedding")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="API para guardar embeddings faciales de aprendices en FaceLit.",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    @app.on_event("startup")
    async def warm_up_model() -> None:
        if not settings.api_key.strip() or settings.api_key.strip() == "change-me":
            raise RuntimeError("API_KEY debe contener una clave de servicio propia antes de iniciar.")
        if not settings.preload_model:
            logger.info("InsightFace startup warm-up disabled")
            return
        logger.info("Warming up InsightFace model")
        get_face_embedding_worker().warm_up()
        logger.info("InsightFace model ready")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        # Pydantic errors may contain the full image/vector as input or a non-JSON exception in ctx.
        errors = [{key: item[key] for key in ("type", "loc", "msg") if key in item} for item in exc.errors()]
        logger.warning(
            "Validation error on %s %s errors=%s",
            request.method,
            request.url.path,
            errors,
        )
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        logger.exception("Unexpected error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno al procesar el registro facial."},
        )

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
