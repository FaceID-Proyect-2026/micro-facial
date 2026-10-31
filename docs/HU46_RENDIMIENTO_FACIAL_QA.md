# FaceLit — Rendimiento de validación facial: revisión para QA

Fecha: 31 de octubre de 2026. Equipo 4. Rama: `hu-46-dev` de micro-facial. HEAD base usado al implementar: `1e12c6c422d3c96ccea4d6103d422e8f2a9c3a3a`. Para la publicación solicitada el 31 de octubre, registrar el SHA de entrega desde el historial de esta rama y la URL del PR; el SHA base no identifica los cambios. La publicación no equivale a una aprobación de QA ni al cierre del problema de rendimiento.

## Problema y alcance

La usuaria reportó más de un minuto en «Captura terminada. Validando el reto…». Una petición de prueba de siete imágenes contra el micro ya arrancado tardó 34,421 s. Esa medición no incluye la captura de cámara ni el recorrido por Java: reproduce un costo importante dentro del procesamiento facial, pero no explica por sí sola cada espera observada en el navegador.

InsightFace ejecutaba cinco modelos por imagen: detección, reconocimiento, dos modelos adicionales de landmarks y edad/género. FaceLit consume el embedding, bbox y cinco puntos de detección; sus métricas de gesto se calculan con el código existente de MediaPipe. Los tres análisis adicionales no alimentaban las decisiones de este flujo. Además, la inferencia síncrona se ejecutaba directamente en rutas/casos de uso asíncronos, bloqueando la atención de otras peticiones mientras calculaba.

Se conserva el flujo de registro y asistencia. No se cambian el frontend, la cámara, los dos retos diferentes, las siete capturas por reto, los umbrales de aceptación, la detección de más de un rostro, la comprobación de identidad ni las consultas de persistencia. No se omiten imágenes ni se reutilizan autorizaciones o resultados biométricos entre solicitudes.

## Archivos y responsabilidades

Rutas relativas a micro-facial:

| Archivo | Cambio |
|---|---|
| `app/workers/face_embedding_worker.py` | Limita InsightFace a detection/recognition; mantiene modelo, tamaño de detección y proveedor CPU. Protege los objetos compartidos con un lock. El precalentamiento ejecuta detección, reconocimiento y preparación de MediaPipe con entradas sintéticas descartadas antes de atender capturas. También retira una importación NumPy sin uso. |
| `app/services/facial_embedding_service.py` | Ejecuta el trabajo de CPU de registro y asistencia mediante `asyncio.to_thread`. La sesión SQL y el acceso al repositorio siguen en la corrutina original. Centraliza la generación de frames y registra solo cantidad y tiempo (`facial inference: frames=... elapsed_ms=...`). |
| `app/api/v1/routes/facial_embeddings.py` | Delega el chequeo de vida síncrono a `run_in_threadpool`; conserva contrato, autenticación y manejo de errores. |
| `app/core/config.py` | Reconoce y valida SERVICE_PORT en el .env compartido con Compose. Corrige el rechazo de ese campo en ejecución directa de Python; no cambia el puerto interno de Uvicorn. |
| `tests/test_inference_runtime.py` | Comprueba selección/caché/precalentamiento de modelos, salud disponible durante inferencia, separación CPU/persistencia y aceptación del puerto de Compose en configuración. |
| `scripts/benchmark_liveness.py` | Petición repetible a liveness-check usando una única cara de la imagen pública t1 incluida en InsightFace. No registra usuarios, rostros ni asistencias. |
| `docs/HU46_RENDIMIENTO_FACIAL_QA.md` | Este informe y pasos de validación. |

Se mantienen workers, servicios, rutas y repositorios en sus carpetas actuales. Se usan asyncio y threading de Python y el threadpool de Starlette ya instalado; no se añaden dependencias ni se modifica PostgreSQL. El lock evita que dos hilos utilicen simultáneamente los objetos de inferencia compartidos; no aumenta la capacidad ilimitadamente y una cola concurrente aún puede agregar espera.

## Preparación y ejecución

1. Usar el ambiente local de QA configurado. La publicación actual es Windows 8091 → contenedor 8090; el backend en Windows usa `FACIAL_EMBEDDING_URL=http://127.0.0.1:8091`. Mantener las claves coincidentes sin publicarlas. La nota del conflicto con Wondershare está en `FaceLit-Backend/docs/HU46_PUERTO_FACIAL_QA.md`.
2. Desde micro-facial, aplicar el código:

```powershell
docker compose up -d --build --no-deps embedding-service
docker compose logs -f embedding-service
```

Compose ya establece `PRELOAD_MODEL=true`. Esperar «Application startup complete»; con el logger de aplicación habilitado a INFO también se muestra «InsightFace model ready». La carga y el precalentamiento pueden prolongar el arranque; no medir una captura mientras el servicio aún no está listo. Un arranque directo con PRELOAD_MODEL deshabilitado conserva costos de inicialización en el primer uso.

3. Abrir `http://127.0.0.1:8091/health` y comprobar `status: ok`. Mantener el backend activo en 8080 y el frontend en 8081.

## Medición reproducible sin datos de usuarios

Desde micro-facial, con el contenedor listo:

```powershell
docker cp scripts/benchmark_liveness.py facelit-embedding-service-embedding-service-1:/tmp/benchmark_liveness.py
docker exec -e PYTHONPATH=/app facelit-embedding-service-embedding-service-1 python -u /tmp/benchmark_liveness.py --label qa --repeat 3
```

El script usa el recorte fijo de una imagen de prueba que trae la biblioteca; no usa las fotos de la usuaria ni descarga nuevas imágenes. Imprime el SHA-256 del recorte, cantidad de frames, duración HTTP y respuesta. El SHA permite comprobar que se comparó la misma entrada; no es un hash de contraseña. La dirección por defecto es el micro dentro del contenedor, `127.0.0.1:8090`.

El resultado correcto es HTTP 200 con `live: false`: las siete imágenes son idénticas y no realizan MOVE_UP. Un 400, error de conexión o `live: true` invalida este ensayo de rendimiento; no debe presentarse como un registro exitoso. La comparación no mide la precisión biométrica ni equivale a una prueba con cámara.

### Resultados observados

| Configuración | Duración de siete imágenes | Resultado |
|---|---|---|
| Antes: cinco modelos, una petición al servicio arrancado | 34,421 s | HTTP 200, captura estática rechazada. |
| Ensayo intermedio: solo modelos usados, sin precalentar inferencia | 23,6638 s; 7,9713 s; 5,9328 s | Tres HTTP 200; misma captura y mismo rechazo. La primera petición conservó costo de preparación. |
| Ajuste final: modelos usados y precalentamiento con entradas sintéticas | 30,6133 s; 6,4432 s; 7,7233 s | Tres HTTP 200; mismo recorte, siete imágenes y mismo rechazo del gesto estático. |

La primera petición final aún tardó 30,6 s: el precalentamiento sintético no eliminó todo el costo del primer análisis de una cara real en este equipo. No se acredita que el problema inicial esté totalmente resuelto ni que el precalentamiento haya acelerado esa primera petición respecto al ensayo intermedio. Las siguientes dos quedaron entre 6,4 y 7,8 s. Estos resultados proceden de una sola máquina y una sola imagen; el antes tiene una observación y no es una distribución estadística. No deben compararse una petición inicial y otra posterior como si fueran un porcentaje general de aceleración. No constituyen un SLA ni garantizan el mismo tiempo con otros equipos, imágenes, redes o carga concurrente.

Los logs están en la carpeta común `ci-evidence/`: `hu46-facial-benchmark-before.log`, `hu46-facial-benchmark-modules-only.log` y, para el ajuste final, `hu46-facial-benchmark-after.log`. El intento anterior que cargaba una segunda instancia del modelo y un recorte rechazado no se usa como comparación de validación.

## Regresión y aceptación con cámara

En un entorno Python de pruebas preparado con los requisitos existentes:

```powershell
python -m pytest -q -rs
python -m ruff check app tests/test_inference_runtime.py scripts/benchmark_liveness.py
```

Resultado del código final antes de reconstruir: 125 pruebas aprobadas, 2 omisiones, 24 advertencias; lint aprobado. Las omisiones son el módulo de decodificación que necesita OpenCV (ausente en el Python de pruebas de Windows) y la integración PostgreSQL que requiere activación explícita. La decodificación JPEG y el rechazo de base64 inválido también se comprobaron directamente con OpenCV real dentro del contenedor. Las reglas de gestos se prueban con señales simuladas; no acreditan precisión con personas reales.

QA debe completar y adjuntar:

- [ ] Rama, SHA final, PR, equipo/CPU y estado del servicio al comenzar.
- [ ] Desde aprendiz: primer reto correcto, segundo diferente y guardado de rostro. Anotar la espera desde «Validando» hasta respuesta en cada reto y el guardado final.
- [ ] Un gesto distinto del solicitado y una captura inmóvil siguen rechazándose. Un fallo no cuenta como acierto y conserva el primer reto válido según el flujo existente.
- [ ] Dos personas en la captura siguen rechazándose; una sola persona no debe fallar por una regresión del cambio.
- [ ] Desde instructor: completar retos y registrar asistencia en una sesión válida; verificar su persistencia y la consulta posterior sin duplicados.
- [ ] Contrastar el tiempo de Network en el navegador y la respuesta del backend; si QA habilita el logger `facelit.embedding` a INFO, comparar también `facial inference` en el micro. No confundir demora de captura, CPU, cola y base de datos.
- [ ] Confirmar que salud/notificaciones pueden responder durante el cálculo; el offload mejora disponibilidad, no elimina la cola de inferencia.
- [ ] Adjuntar las ejecuciones de pruebas y CI del SHA entregado, sin imágenes de usuarios, embeddings, claves ni JWT.

Estado: implementación local y verificación automatizada; primera validación aún lenta, aceptación E2E con cámara y revisión independiente pendientes. No se declara cerrado el rendimiento de todos los flujos únicamente por mejorar el ensayo de una captura estática. QA debe tratar la espera de primera validación como pendiente de rendimiento, no aprobarla por los tiempos de las peticiones posteriores.
