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

La configuración se entrega para publicación en hu-02-dev y revisión mediante PR hacia dev. La ejecución verde en GitHub Actions y la revisión independiente quedan pendientes de verificar. No declarar la HU Done sin esas evidencias.
