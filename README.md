# FaceLit Embedding Service

Microservicio FastAPI para registrar y consultar embeddings faciales de aprendices durante el registro facial.

## Requisitos

- Python 3.11+
- PostgreSQL con las migraciones de `FaceLit-DB`
- Para usar InsightFace en Windows: Microsoft Visual C++ Build Tools 14.0+

## Configuracion

Copia `.env.example` a `.env` y ajusta las variables:

```env
DATABASE_URL=postgresql+asyncpg://facelit_user:facelit_password@localhost:5439/facelit-db
API_KEY=change-me
```

## Ejecutar

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8090
```

En Windows, si `pip install -r requirements.txt` falla instalando `insightface` con un mensaje de Microsoft Visual C++ 14.0, instala "Microsoft C++ Build Tools" y repite la instalacion. Con Docker, el `Dockerfile` ya incluye herramientas de compilacion para Linux.

OpenAPI queda disponible en:

- `http://localhost:8090/docs`
- `http://localhost:8090/openapi.json`

## Endpoints principales

- `POST /api/v1/facial-embeddings` guarda un embedding facial activo por aprendiz.
- `POST /api/v1/facial-embeddings/from-image` genera el embedding con InsightFace desde una imagen y lo guarda.
- `GET /api/v1/facial-embeddings/users/{user_id}` consulta el embedding activo.
- `DELETE /api/v1/facial-embeddings/users/{user_id}` desactiva el embedding activo.
- `GET /health` verifica el estado del servicio.

Incluye `X-API-Key` en las peticiones cuando `API_KEY` tenga valor.

## Generar embedding desde imagen

`POST /api/v1/facial-embeddings/from-image`

```json
{
  "user_id": "uuid-del-aprendiz",
  "image_base64": "data:image/jpeg;base64,...",
  "photo_reference": "capture://registro-aprendiz.jpg",
  "replace_existing": false,
  "created_by": "mobile-app"
}
```

El endpoint detecta exactamente un rostro con InsightFace, genera el vector facial y lo persiste en `facialrecognition.user_face`.
