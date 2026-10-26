# HU-MEJ-006 — CI del microservicio

## Alcance

Configuración en hu-02-dev, basada en 911f124ea4665bbe4345b9f38a113ddb235072d3. No se cambian endpoints, reconocimiento facial, modelos, reglas de vida ni pruebas existentes. requirements-ci.txt incluye requirements-dev.txt y añade herramientas de CI con versión fijada; requirements.txt y el Dockerfile de producción no cambian.

## Qué ejecuta GitHub Actions

Archivo .github/workflows/ci.yml: todos los PR, pushes a dev/qa/main y ejecución manual.

1. Preparar Python 3.11 y las bibliotecas nativas usadas por el proyecto.
2. Instalar requirements-ci.txt y verificar compatibilidad con pip check.
3. Ejecutar Ruff para errores de sintaxis, nombres no definidos y errores básicos (E9,F63,F7,F82). No se obliga a reformatear el código existente.
4. Compilar la sintaxis de app y tests mediante compileall.
5. Ejecutar pytest con pytest-cov y conservar resumen, JUnit XML, cobertura XML y HTML.

Artifact: micro-facial-tests-and-coverage. Abrir htmlcov/index.html para explorar líneas cubiertas. No hay umbral mínimo acordado; el pipeline falla si fallan lint, instalación o pruebas, pero no por una meta de cobertura inventada.

## Aislamiento y límites

CI define valores ficticios de API_KEY y DATABASE_URL, environment ci y preload_model false. No necesita credenciales reales ni fotos de producción. Las pruebas existentes usan fixtures y workers falsos; no acreditan reconocimiento real de una persona.

tests/test_face_reset_postgres.py requiere RF54_POSTGRES_TEST=1 y PostgreSQL con esquema preparado. Permanece omitido en el job estándar; pytest -ra muestra explícitamente la omisión. No se presenta como aceptación de PostgreSQL ni como E2E.

## Protección y evidencia

Tras publicar y ejecutar el PR, exigir el check Python unit tests and coverage en el ruleset de las ramas de integración, además de PR y aprobación independiente. Registrar SHA, PR, run y artifact.

## Validación local ejecutada

- YAML parseado correctamente y git diff --check aprobado.
- Python 3.11 en un contenedor temporal, con app y tests montados de solo lectura, sin .env ni conexiones productivas.
- pip check: sin dependencias incompatibles.
- Ruff E9,F63,F7,F82: sin errores.
- Sintaxis Python: 25 archivos analizados sin errores.
- pytest: 27 aprobadas, 1 omitida (PostgreSQL opt-in), 4 advertencias; salida 0.
- Cobertura de líneas: 48,33 %, medida con pytest-cov; no equivale a cobertura completa ni valida biometría real.
- Evidencia local: ../ci-evidence/micro/run.log, ../ci-evidence/micro/pytest-results.xml, ../ci-evidence/micro/coverage.xml y ../ci-evidence/micro/htmlcov/index.html. Son salidas locales, no archivos de código para el commit.

Esta prueba usa dependencias de una imagen existente y no sustituye una instalación limpia en GitHub Actions. El job de CI instala todas las dependencias desde requirements-ci.txt.

## Actualización de evidencia y alcance — 25 de octubre de 2026

Rama de esta entrega: `hu-02-dev`, PR [#4 hacia dev](https://github.com/FaceID-Proyect-2026/micro-facial/pull/4). El SHA base `9e60c34f70875b9bff470d5313b4a90609804ab0` ya tiene una [ejecución real verde](https://github.com/FaceID-Proyect-2026/micro-facial/actions/runs/40195796057). Al publicar esta actualización documental debe comprobarse el CI del nuevo head del PR.

Se repitieron dependencias CI, pip check, Ruff E9/F63/F7/F82, compileall y pytest con Python 3.11.17 sobre una copia del código versionado. Resultado habitual: **27 aprobadas, 1 omitida, 4 advertencias de deprecación, salida 0**, cobertura **48,33 % de líneas (406/840)**.

Además se ejecutó aparte `RF54_POSTGRES_TEST=1 python -m pytest tests/test_face_reset_postgres.py` contra PostgreSQL temporal: **1 aprobada, 0 omitidas**. Se verificaron registro/reemplazo, consumo de autorización, rechazo de reutilización/expiración y rollback con vectores sintéticos. Entre ambas invocaciones se ejercitaron los 28 casos. La cobertura anterior corresponde solo a la suite habitual; no se sumó la ejecución adicional ni se afirma que el workflow actual ejecute el opt-in.

La base temporal se eliminó sin tocar desarrollo. Se usó una imagen local con dependencias nativas existentes; no es evidencia de una construcción desde cero. Logs, XML e informes HTML están fuera del repositorio en `ci-evidence/hu006-validation-20261025/` del directorio de trabajo. No se validó biometría real.

La delimitación solicitada asigna HU-MEJ-006 a **H-16/H-18: automatización, cobertura y evidencia**. Implementar reglas de seguridad H-17 corresponde a HU-MEJ-001/HU-MEJ-002; CI debe ejecutar sus pruebas cuando se integren. La propuesta inicial citaba H-17 y esta aclaración conserva su trazabilidad.

Se configuró como obligatorio `Python unit tests and coverage` en el ruleset CI activo de dev/qa/main, con una aprobación y sin bypass. La responsable informa haber probado el bloqueo con CI rojo: falta adjuntar y contrastar su evidencia, junto con las aprobaciones independientes del PR. La cobertura menor al 100 % identifica código no recorrido, no un fallo por sí misma ni una garantía sobre partes sin probar. No se cambiaron fuentes ni workflow en esta actualización documental. No declarar Done antes de evidencia y aprobación.

Documento de entrega compartido: [alcance, inventario y resultados de HU-MEJ-006](https://github.com/FaceID-Proyect-2026/FaceLit/blob/hu-38-dev/docs/HU-MEJ-006-ALCANCE-Y-ENTREGA-QA.md).
