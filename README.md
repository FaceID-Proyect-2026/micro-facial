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
uvicorn app.main:app --port 8090
```

En Windows, si `pip install -r requirements.txt` falla instalando `insightface` con un mensaje de Microsoft Visual C++ 14.0, instala "Microsoft C++ Build Tools" y repite la instalacion. Con Docker, el `Dockerfile` ya incluye herramientas de compilacion para Linux.

OpenAPI queda disponible en:

- `http://localhost:8090/docs`
- `http://localhost:8090/openapi.json`

## Endpoints principales

- `POST /api/v1/facial-embeddings` guarda un embedding facial activo por aprendiz.
- `POST /api/v1/facial-embeddings/from-image` genera el embedding con InsightFace desde una imagen y lo guarda.
- `POST /api/v1/facial-embeddings/verify-session` genera un embedding temporal desde una imagen, lo compara contra los aprendices activos de la ficha de `record_environment` y responde si coincide.
- `GET /api/v1/facial-embeddings/users/{user_id}` consulta el embedding activo. `user_id` puede ser `academic.apprentice.id_apprentice` o `academic.apprentice.id_user_app`.
- `PATCH /api/v1/facial-embeddings/users/{user_id}` actualiza parcialmente el embedding activo. `user_id` puede ser `academic.apprentice.id_apprentice` o `academic.apprentice.id_user_app`.
- `DELETE /api/v1/facial-embeddings/users/{user_id}` desactiva el embedding activo. `user_id` puede ser `academic.apprentice.id_apprentice` o `academic.apprentice.id_user_app`.
- `GET /health` verifica el estado del servicio.

Incluye `X-API-Key` en las peticiones cuando `API_KEY` tenga valor.

## Generar embedding desde imagen

`POST /api/v1/facial-embeddings/from-image`

```json
{
  "user_id": "uuid-del-aprendiz-o-del-usuario",
  "image_base64": "data:image/jpeg;base64,...",
  "photo_reference": "capture://registro-aprendiz.jpg",
  "replace_existing": false,
  "created_by": "mobile-app"
}
```

El endpoint detecta exactamente un rostro con InsightFace, genera el vector facial y lo persiste en `facialrecognition.user_face.id_apprentice`.

## Verificar rostro temporal en una sesion

`POST /api/v1/facial-embeddings/verify-session`

```json
{
  "record_environment_id": "uuid-de-la-sesion",
  "image_base64": "data:image/jpeg;base64,...",
  "threshold": 0.65
}
```

El microservicio genera el embedding solo en memoria y consulta los candidatos con esta ruta:

```text
environment.record_environment -> academic.apprentice_chip -> facialrecognition.user_face
```

Solo compara contra aprendices activos de la ficha asociada a la sesion activa. La respuesta esta pensada para que el backend decida si crea `facialrecognition.facial_event`:

```json
{
  "match": true,
  "id_apprentice": "uuid-del-aprendiz",
  "similarity": 0.82,
  "threshold": 0.65,
  "model_name": "insightface/buffalo_l",
  "reason": "MATCH_FOUND"
}
```

Si `match` es `false`, el backend no debe crear asistencia aprobada y puede responder al frontend con "rostro no coincide".

## Actualizar parcialmente un embedding

`PATCH /api/v1/facial-embeddings/users/{user_id}`

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "photo_reference": "capture://registro-aprendiz-actualizado.jpg",
  "updated_by": "mobile-app"
}
```

Tambien puede enviarse un vector ya calculado:

```json
{
  "embedding": [
    0.012, 0.034, 0.056, 0.078, 0.091, 0.023, 0.045, 0.067,
    0.089, 0.011, 0.032, 0.054, 0.076, 0.098, 0.021, 0.043,
    0.065, 0.087, 0.019, 0.031, 0.053, 0.075, 0.097, 0.029,
    0.041, 0.063, 0.085, 0.017, 0.039, 0.051, 0.073, 0.095
  ],
  "model_name": "facenet-test",
  "photo_reference": "capture://registro-aprendiz-actualizado.jpg",
  "updated_by": "mobile-app"
}
```

Todos los campos son opcionales salvo que debe enviarse al menos uno entre `image_base64`, `embedding`, `model_name` o `photo_reference`. Si se envia `image_base64`, el servicio genera el embedding con InsightFace. Si se envia `embedding`, el servicio recalcula `embedding_dimension`.


# Comando de ejecucion local

```
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8090
```

En Windows evita `--reload` si aparece `PermissionError: [WinError 5] Acceso denegado`;
ese error viene del subproceso de recarga de Uvicorn, no del servicio.

# En pc propio
```
APP_NAME=FaceLit Embedding Service
ENVIRONMENT=local
API_KEY=change-me
DATABASE_URL=postgresql+asyncpg://facelit_user:facelit_password@localhost:5439/facelit-db
CORS_ORIGINS=["http://localhost:8081","http://localhost:19006","http://localhost:3000"]
MIN_EMBEDDING_DIMENSION=32
MAX_EMBEDDING_DIMENSION=4096
```

# comando para ejecutar el contenedor
```
docker compose up -d --build       
```

# En contenedor
```
APP_NAME=FaceLit Embedding Service
ENVIRONMENT=local
API_KEY=change-me
DATABASE_URL=postgresql+asyncpg://facelit_user:facelit_password@postgres:5432/facelit-db
CORS_ORIGINS=["http://localhost:8081","http://localhost:19006","http://localhost:3000"]
MIN_EMBEDDING_DIMENSION=32
MAX_EMBEDDING_DIMENSION=4096
```

# Docs
```
http://localhost:8090/docs
```
