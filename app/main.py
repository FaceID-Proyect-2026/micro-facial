import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings

logger = logging.getLogger("facelit.embedding")


def _summarize_body(body):
    if not isinstance(body, dict):
        return body

    summary = dict(body)
    image = summary.get("image_base64")
    if isinstance(image, str):
        summary["image_base64"] = f"<base64 length={len(image)} prefix={image[:32]!r}>"
    photo_reference = summary.get("photo_reference")
    if isinstance(photo_reference, str) and len(photo_reference) > 120:
        summary["photo_reference"] = (
            f"<photo_reference length={len(photo_reference)} prefix={photo_reference[:32]!r}>"
        )
    return summary


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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        logger.warning(
            "Validation error on %s %s errors=%s body=%s",
            request.method,
            request.url.path,
            exc.errors(),
            _summarize_body(exc.body),
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
