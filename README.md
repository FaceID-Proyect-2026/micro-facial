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
| `CORS_ORIGINS` | Explicit browser-origin configuration |
| `PRELOAD_MODEL` | Optional startup model warm-up |
| `INSIGHTFACE_MODEL_NAME`, `INSIGHTFACE_DET_SIZE`, `INSIGHTFACE_CTX_ID` | Model and execution configuration |
| `MIN_EMBEDDING_DIMENSION`, `MAX_EMBEDDING_DIMENSION` | Allowed vector dimensions |

Inspect [config.py](app/core/config.py) for current defaults and liveness thresholds. `API_KEY` must be a service-specific secret: application startup fails when it is empty or set to `change-me`.

## API and flow contracts

OpenAPI is available at `/openapi.json`, with `/docs` and `/redoc`, only when
`ENVIRONMENT=local`. In production, staging and CI these HTTP routes return 404;
the default environment is production. Exact schemas remain defined in
[facial_embedding.py](app/schemas/facial_embedding.py) and
[the router](app/api/v1/routes/facial_embeddings.py).

`CORS_ORIGINS` accepts an explicit CSV or JSON list of HTTP/HTTPS browser origins.
An empty list is the default and is appropriate when only the backend calls this
private service. Wildcards, origin regexes and browser credentials are not enabled.
Configure exact local origins when browser access is intentionally required.
Backend-to-micro HTTP calls and `X-API-Key` authentication are unaffected by CORS;
capture, model caching and inference parameters are unchanged.

| Method and path | Purpose |
| --- | --- |
| `POST /api/v1/facial-embeddings` | Register a supplied vector under existing authorization rules |
| `POST /api/v1/facial-embeddings/from-image` | Generate/register from capture and optional liveness frames |
| `POST /api/v1/facial-embeddings/liveness-check` | Evaluate a requested liveness challenge |
| `POST /api/v1/facial-embeddings/verify-session` | Match capture against eligible session candidates |
| `GET /api/v1/facial-embeddings/apprentices/{apprentice_id}` | Read an active record by apprentice ID |
| `PATCH /api/v1/facial-embeddings/apprentices/{apprentice_id}` | Update permitted metadata; changing vector/image is rejected |
| `DELETE /api/v1/facial-embeddings/apprentices/{apprentice_id}` | Deactivate an active record by apprentice ID |
| `GET /api/v1/facial-embeddings` | List active summaries with pagination |
| `GET /health` | Process/startup health response |

Requests protected by the configured internal key send `X-API-Key`. FaceLit-Backend resolves the authenticated account to its active apprentice record and passes the `apprentice_id` and reset owner to this service. Micro-facial no longer queries identity, academic or session tables; it reads and writes the facial-recognition schema. Replacing an existing template requires the approved, unexpired, single-use reset authorization and the live-capture path; failed writes roll back authorization consumption. For attendance, the backend validates the session and sends only the active apprentice IDs in `candidate_apprentice_ids`; this service compares against active facial templates in that bounded list.

The response schemas expose metadata and match results, not embedding vectors. The database is still shared operationally; configure a micro-facial database account with grants limited to the facial-recognition objects before claiming database-level isolation.

## Database account boundary

Apply the FaceLit-DB changelog, then provision the dedicated login with
`FaceLit-DB/scripts/provision_micro_facial_login.ps1` from the FaceLit-DB
repository (the repositories are versioned independently).
The operator supplies `FACELIT_MICRO_DB_PASSWORD` through the deployment secret
store or process environment. Configure `DATABASE_URL` with this login; do not
reuse the backend or migration account. The group role can read/write facial
templates and consume approved reset tokens, and has no grants on identity,
academic, environment, attendance-event, or notification tables.

The database remains physically shared. This account boundary limits what the
microservice can access at PostgreSQL, while FaceLit-Backend remains the source
of identity, session and apprentice eligibility decisions.

## Docker and model storage

```bash
docker compose build embedding-service
docker compose up -d embedding-service
docker compose logs --tail=80 embedding-service
```

The standalone Compose file expects the external `facelit-docker-compose_default` database network. Its internal database hostname is `postgres`; host processes normally use localhost with the database's configured host port. The Compose service is named `embedding-service`, while a workspace-level deployment may use another private alias.

The Dockerfile uses a build stage for native dependencies and a smaller runtime stage. The process runs as UID/GID 10001 and reports `/health` through Docker `HEALTHCHECK`. Compose stores models in `/home/facelit/.insightface`; initial model download requires network access and storage.

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

### Container health and model cache

The Docker build installs Python dependencies in a build stage. The runtime
contains the virtual environment and native runtime libraries, without
`build-essential`, and runs as UID/GID `10001:10001`. Docker checks `/health`
every 30 seconds, with a 300-second startup grace period. This process probe does
not run recognition and does not certify database connectivity or camera accuracy.
It does not add a delay to any registration/attendance request.

The existing `insightface-models` volume now mounts at
`/home/facelit/.insightface`; the model loader and preloading policy are unchanged.
Keep the same Compose project/volume to reuse downloaded models. For an existing
root-owned cache, the operator must grant its new owner write access once before
starting the new image (after building it, while the service is stopped):

```bash
docker compose run --rm --no-deps --user root --entrypoint chown embedding-service -R 10001:10001 /home/facelit/.insightface
```

This maintenance command changes ownership of the existing model cache; it does
not remove the volume or access PostgreSQL. New volumes receive the directory
ownership from the image. A fresh or missing cache still requires model download;
the healthcheck does not trigger downloads. Any external deployment Compose must
use the same updated mount path and compatible ownership. First startup/download
time is separate from request latency, which must be measured with a real camera.

Only four liveness challenges are enabled: `MOVE_UP`, `MOVE_DOWN`, `MOVE_CLOSER`
and `MOVE_AWAY`. Registration and attendance still require two distinct challenges.
The frontend selects only these four, and the microservice rejects the seven retired
codes even when an older client submits them directly. Their detection thresholds
and capture timing are unchanged.

`tests/test_all_liveness_contracts.py` covers complete signals, all twelve incorrect
substitutions, all twelve distinct pairs, still faces, identity changes and invalid
measurements. The authenticated HTTP integration checks the four active challenges
and rejection of the seven retired codes. Synthetic inference verifies rules and
transport; physical-camera acceptance remains a separate check. Never use the test
integration entry point in production.

Record the exact component commits, CI run and artifacts for a coordinated release. Verify private connectivity, credentials, model startup, authorized registration/reset and attendance with the deployed backend/frontend. Obtain independent review; do not reuse a green result from another commit as the final head's CI evidence.

All maintained explanatory documentation is consolidated in this README. The previous CI narrative remains recoverable from Git history and the separate delivery archive. Python code, tests, workflow YAML, requirements, configuration examples, Docker inputs and source comments remain in their functional locations.

Related repositories: [backend](https://github.com/FaceID-Proyect-2026/FaceLit-Backend), [frontend](https://github.com/FaceID-Proyect-2026/FaceLit), [database](https://github.com/FaceID-Proyect-2026/FaceLit-DB).

## Revision-tagged container release (HU-48)

The manual `.github/workflows/container-release.yml` first runs this repository's
reusable CI. Only a successful result permits building/publishing to GHCR. The
artifact `container-release-reference` records the source revision and registry
digest. Publication is an explicit workflow dispatch; no image was published as
part of the local implementation. Publish the coordinated database branch before
dispatching CI that references it. Deployment and rollback use recorded digests
and a compatible database schema; see the workspace `deploy/README.md` and the
HU-48 assessment in facelit-docs. A passing local test is not production acceptance.

## Application dependency boundary (HU-48)

The application service now requires an explicit generator and repository. Its
transaction dependency is described by UnitOfWork (commit/rollback), not an
SQLAlchemy class. api/dependencies.py composes the production implementations;
the E2E fixture composes its real repository and deterministic inference worker.
GeneratedEmbedding is a plain value in app/domain/embedding.py, re-exported by
the worker module for existing callers. HTTP routes, schemas, SQL, model behavior,
thread ownership and inference caching are preserved. Internal callers that
construct the service must pass a repository explicitly.
