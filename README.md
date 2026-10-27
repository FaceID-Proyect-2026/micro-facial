# FaceLit facial embedding service

FastAPI service for facial-template registration, liveness checks and attendance-session matching. It integrates with the FaceLit backend and PostgreSQL schema managed by [FaceLit-DB](https://github.com/FaceID-Proyect-2026/FaceLit-DB).

## Architecture

Three logical layers are kept: HTTP routes and schemas, application services/rules,
and SQL/model adapters. `api/dependencies.py` composes request dependencies so routes
do not instantiate persistence or inference implementations. `FacialEmbeddingService`
remains the compatible application facade; `LivenessValidator` owns temporal rules,
and small protocols describe the generator, policy and store it consumes.

The backend HTTP integration and shared PostgreSQL schema are preserved. The fixture
in `tests/support/e2e_app.py` uses the real routes, rules and database but substitutes
image inference. It is exclusively for the sibling DB repository's isolated E2E
runner and must never be used as the deployment entry point. Production still uses
`app.main:app` and the cached real worker.

```text
HTTP routes -> service rules -> repository / PostgreSQL
                    |
             embedding worker
          InsightFace / MediaPipe / ONNX

app/api/           Router, dependencies and access guard
app/core/          Environment/settings model
app/db/            Async SQLAlchemy sessions
app/repositories/  Current SQL persistence and matching queries
app/schemas/       Pydantic request/response contracts
app/services/      Registration, authorization and liveness coordination
app/workers/       Image decoding, models and facial metrics
tests/             Automated checks and opt-in PostgreSQL integration
.github/workflows/ Existing lint/test/coverage workflow
README.md          Single maintained documentation entry
```

The repository resolves apprentice/user relationships and environment-session candidates through the current FaceLit schemas. It does not yet have an independent database ownership boundary. Such an architectural migration, and the separate credential-hardening story, are not implemented by this documentation consolidation.

## Requirements and setup

Use the Python 3.11 container baseline and the versions in [requirements.txt](requirements.txt). The declared runtime includes FastAPI 0.115.6, SQLAlchemy 2.0.36, asyncpg 0.30.0, InsightFace 0.7.3, MediaPipe 0.10.14, NumPy 2.2.1 and ONNX Runtime 1.20.1. Native image/model dependencies require the appropriate compiler/runtime libraries when installing outside Docker; Windows installation of InsightFace may require Microsoft C++ Build Tools.

```bash
python -m venv .venv
```

Activate `.venv` for your shell (`.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Unix), then:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --port 8090
```

Copy [.env.example](.env.example) only if no local `.env` exists, and set the intended private environment values. For a list setting such as CORS, use a JSON array in the local environment, for example `CORS_ORIGINS=["http://localhost:8081"]`; the inherited example's comma-separated list may require this adjustment with the current settings parser.

| Setting | Purpose |
| --- | --- |
| `DATABASE_URL` | Async SQLAlchemy PostgreSQL URL for the target FaceLit database |
| `API_KEY` | Internal credential matching backend requests |
| `CORS_ORIGINS`, `CORS_ORIGIN_REGEX` | Browser-origin configuration |
| `PRELOAD_MODEL` | Optional startup model warm-up |
| `INSIGHTFACE_MODEL_NAME`, `INSIGHTFACE_DET_SIZE`, `INSIGHTFACE_CTX_ID` | Model and execution configuration |
| `MIN_EMBEDDING_DIMENSION`, `MAX_EMBEDDING_DIMENSION` | Allowed vector dimensions |

Inspect [config.py](app/core/config.py) for current defaults and liveness thresholds. Configure real values explicitly. In this source revision, authentication is conditional on a nonempty `API_KEY`; an empty value is not suitable for a deployed service. Mandatory credentials and removal of development fallbacks belong to their coordinated security/container deliveries, not this README-only change.

## API and flow contracts

OpenAPI is available locally at `/openapi.json`, with interactive documentation at `/docs` and `/redoc`. Exact schemas are defined in [facial_embedding.py](app/schemas/facial_embedding.py) and [the router](app/api/v1/routes/facial_embeddings.py).

| Method and path | Purpose |
| --- | --- |
| `POST /api/v1/facial-embeddings` | Register a supplied vector under existing authorization rules |
| `POST /api/v1/facial-embeddings/from-image` | Generate/register from capture and optional liveness frames |
| `POST /api/v1/facial-embeddings/liveness-check` | Evaluate a requested liveness challenge |
| `POST /api/v1/facial-embeddings/verify-session` | Match capture against eligible session candidates |
| `GET /api/v1/facial-embeddings/users/{user_id}` | Read an active record |
| `PATCH /api/v1/facial-embeddings/users/{user_id}` | Update permitted metadata; changing vector/image is rejected by the current service |
| `DELETE /api/v1/facial-embeddings/users/{user_id}` | Deactivate an active record |
| `GET /api/v1/facial-embeddings` | List active summaries with pagination |
| `GET /health` | Process/startup health response |

Requests protected by the configured internal key send `X-API-Key`. Enrollment resolves the supplied user/apprentice to the current academic owner. Replacing an existing template requires the existing approved, unexpired, single-use reset authorization and the live-capture path; failed writes must roll back authorization consumption. Liveness compares frames/challenges and facial metrics. Session verification resolves eligible candidates before returning match status; the backend remains responsible for its attendance flow.

Do not infer that this documentation change removes biometric fields from the existing response models or completes network isolation. Review and deploy the separately coordinated security change before claiming those acceptance criteria.

## Docker and model storage

```bash
docker compose build embedding-service
docker compose up -d embedding-service
docker compose logs --tail=80 embedding-service
```

The standalone Compose file expects the external `facelit-docker-compose_default` database network. Its internal database hostname is `postgres`; host processes normally use localhost with the database's configured host port. The Compose service is named `embedding-service`, while a workspace-level deployment may use another private alias.

The current Dockerfile is Python 3.11 with native build/runtime libraries, and the current Compose model volume mounts at `/root/.insightface`. Model warm-up/download can take time and requires network access and storage. Non-root image construction, model-volume permission changes and image health probes are part of HU-MEJ-015, preserved separately and not silently applied here.

The project owner identified Hostinger `srv2068863.hstgr.cloud` as the deployment target. Keep the facial service private to the intended backend/network; it does not require a public URL. A public frontend page returning HTTP 200 does not verify this service or its model.

## Tests and CI

Install [requirements-dev.txt](requirements-dev.txt) for development tests, or [requirements-ci.txt](requirements-ci.txt) for the existing CI lint/coverage tools. Use fictitious local settings and disable model preloading for isolated tests.

```bash
python -m pip install -r requirements-ci.txt
python -m pip check
python -m ruff check --select E9,F63,F7,F82 app tests
python -m compileall -q app tests
python -m pytest -q -ra --cov=app --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=html:htmlcov --junitxml=pytest-results.xml
```

[ci.yml](.github/workflows/ci.yml) runs the `Python unit tests and coverage` job for PRs, configured integration-branch pushes and manual dispatch. It installs dependencies, runs compatibility/lint/syntax checks and publishes the `micro-facial-tests-and-coverage` artifact. No coverage threshold was agreed in that workflow; a coverage percentage is not proof of full biometrics acceptance.

The opt-in PostgreSQL test requires `RF54_POSTGRES_TEST=1` and a prepared local database. It refuses remote hosts and uses a rollback-contained fixture; only enable it against a suitable verification environment. A standard run can legitimately show that test skipped. Mock workers and image-decoding tests do not certify a physical camera, model download or a real person's recognition.

## Release and documentation policy

Record the exact component commits, CI run and artifacts for a coordinated release. Verify private connectivity, credentials, model startup, authorized registration/reset and attendance with the deployed backend/frontend. Obtain independent review; do not reuse a green result from another commit as the final head's CI evidence.

All maintained explanatory documentation is consolidated in this README. The previous CI narrative remains recoverable from Git history and the separate delivery archive. Python code, tests, workflow YAML, requirements, configuration examples, Docker inputs and source comments remain in their functional locations.

Related repositories: [backend](https://github.com/FaceID-Proyect-2026/FaceLit-Backend), [frontend](https://github.com/FaceID-Proyect-2026/FaceLit), [database](https://github.com/FaceID-Proyect-2026/FaceLit-DB).
