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

## Seguimiento: asistencia de 3 minutos y 35 segundos

La usuaria reportó 215 segundos para un aprendiz. Es una observación del flujo completo, no un tiempo medido aisladamente de SQL o red. Se conserva como incidencia abierta hasta repetir el recorrido con cámara y adjuntar tiempos.

### Diagnóstico y cambios acotados

- El cliente actual valida siete frames por reto y vuelve a enviar los catorce al guardar. Antes el worker volvía a calcular las características de esos mismos frames. Se conserva la validación final; se reutiliza únicamente su inferencia cuando la entrada es exactamente la misma.
- `app/workers/face_embedding_worker.py`: caché en memoria del worker, bajo su lock existente, indexada por SHA-256 de la entrada. Límite predeterminado 64 resultados, validez máxima de reutilización 60 segundos, sin extenderla con los accesos. No almacena imágenes, identificadores de usuarios, permisos, resultados de liveness ni decisiones de asistencia. Sí contiene temporalmente características biométricas derivadas. Las entradas caducadas se purgan al procesar otra captura; es caducidad de reutilización, no una tarea de borrado por reloj durante inactividad. El proceso las pierde al reiniciarse y la capacidad permanece limitada. No es persistencia en PostgreSQL.
- La consulta actual de candidatos, el umbral, la cuenta activa, la sesión y las reglas de ambos retos se vuelven a comprobar por sus capas existentes. Errores de inferencia no se almacenan. Al expirar o desalojar una entrada se recalcula; esto puede tardar más, pero no concede acceso.
- `app/core/config.py` y `.env.example`: `INFERENCE_CACHE_SIZE` y `INFERENCE_CACHE_TTL_SECONDS`, con límites y opción de desactivar mediante 0. No es necesario editar el `.env` privado para usar los valores predeterminados.
- El calentamiento anterior con imagen vacía no ejercitaba la rama de landmarks de FaceMesh sobre una cara. El calentamiento actual agrega un recorte de la imagen pública `t1` incluida en InsightFace 0.7.3. Se usa antes de completar startup, se descarta y no se registra ni introduce en la caché de capturas. Puede aumentar la espera de arranque a cambio de preparar esa rama antes de atender al primer usuario. No convierte la fotografía de prueba en una prueba de presencia real.
- Se confirmó en el código de InsightFace 0.7.3 que `allowed_modules` filtra después de construir las sesiones ONNX. El cargador de `buffalo_l` ahora abre directamente los mismos `det_10g.onnx` y `w600k_r50.onnx`, conserva los métodos `get/prepare` de FaceAnalysis y verifica el tipo de cada modelo. Evita cargar para luego descartar modelos de edad/género y landmarks no usados. Los otros paquetes mantienen su cargador anterior. No se aplicó el ajuste experimental de hilos. Este adaptador depende de los nombres del paquete fijado: una actualización de InsightFace/modelos requiere revisar la prueba y el cargador.
- `tests/test_frame_inference_cache.py`: siete pruebas sobre reutilización exacta, gesto distinto rechazado, expiración fija, capacidad/LRU, desactivación por tamaño o TTL, errores y concurrencia. `tests/test_inference_runtime.py` agrega comprobación de ambas ramas de calentamiento con dependencias simuladas.
- `docker-compose.yml`: `init: true` para gestionar señales y recoger procesos al detener el servicio. Durante la aplicación local Docker informó un PID zombie e impidió reemplazar el contenedor; se retiró únicamente ese contenedor con `docker rm -f`, conservando el volumen de modelos. No se reinició Docker Desktop ni PostgreSQL. Este incidente de operación no demuestra por sí solo la causa de los 215 segundos originales.
- `scripts/benchmark_liveness.py --flow`: genera catorce codificaciones JPEG diferentes de la imagen pública, mide dos grupos nuevos de siete y su revalidación de catorce. Usa solo `/liveness-check`; no escribe rostros ni asistencia. El cliente lee la imagen de la dependencia sin importar otra instancia completa de InsightFace para no competir innecesariamente por memoria. Las capturas estáticas deben seguir devolviendo `live: false`.

No se cambian los dos retos, sus umbrales, la resolución, el modelo, endpoints, SQL, esquema ni lógica de cámara. No se agregan dependencias ni se modifica la arquitectura por capas. Frontend y backend no requieren cambios funcionales para aprovechar esta reutilización.

### Observaciones y límites de las mediciones

Una petición HTTP estática anterior al cambio no respondió dentro de 180 segundos. La consulta a `pg_stat_activity` no mostró consultas activas bloqueadas en ese instante; eso no excluye problemas de BD en otros momentos. No se atribuye toda la demora a la conexión entre APIs.

Se ensayó por separado una configuración de 1 y 2 hilos con el mismo modelo. Hubo gran variación entre carga, primera inferencia y repeticiones; no produjo una mejora consistente y no se incorpora ese adaptador al producto. La ejecución auxiliar usaba otra instancia y competía por recursos; no sirve como comparación justa de la latencia del usuario. Durante ese diagnóstico, Windows reportó aproximadamente 0,5 GB libres de 8 GB; no se presenta como prueba de que la RAM sea la única causa del reporte original.

Regresión final del worker/configuración y cargador selectivo: **134 aprobadas, 2 omitidas, 24 advertencias**; lint aprobado. Las omisiones siguen siendo OpenCV ausente en el Python de pruebas Windows y la integración PostgreSQL opt-in. El calentamiento y el cargador se prueban unitariamente con dependencias simuladas; su ejecución real se comprueba al arrancar el contenedor. Evidencia: `ci-evidence/hu46-cache-regression.log`, en la carpeta común.

### Repetición del ensayo HTTP sin datos de usuarios

Desde micro-facial, después de que el servicio termine startup y `/health` responda:

```powershell
docker cp scripts/benchmark_liveness.py facelit-embedding-service-embedding-service-1:/tmp/benchmark_liveness.py
docker exec -e PYTHONPATH=/app facelit-embedding-service-embedding-service-1 python -u /tmp/benchmark_liveness.py --flow --label hu46-cache --repeat 1
```

Las tres respuestas deben ser HTTP 200 y `live: false`. El último paso reutiliza las mismas entradas, pero este endpoint vuelve a ejecutar sus reglas y debe rechazar la foto estática. No es un registro completo ni una medida de cámara, búsqueda SQL o persistencia. No comparar este modo con el ensayo histórico de siete copias idénticas como si fueran la misma carga.

Pendiente: registrar la medición del servicio reconstruido y repetir entrada/salida con cámara real para dos aprendices consecutivos, con el servicio ya preparado. Anotar por separado captura, validación de cada reto y guardado; conservar el UUID de sesión y los eventos resultantes. No se declara corregido todo el tiempo reportado sin esa aceptación.

### Perfil de la espera y corrección de propiedad del detector

El ensayo con siete codificaciones nuevas volvió a superar 180 segundos incluso con el servicio preparado. Por tanto, la caché y el cargador selectivo no bastaban para cerrar la incidencia. Se usó temporalmente [py-spy](https://github.com/benfred/py-spy) 0.4.2, fuera de la imagen de producción y sin `--locals`. La muestra de pila dejó el loop HTTP principal en `select` y el hilo AnyIO de validación esperando en `mediapipe/python/solution_base.py:340`, llamado desde `FaceMesh.process`, no en SQL. Evidencia local: `ci-evidence/hu46-mediapipe-profile.log`. El contenedor auxiliar fue retirado; no se agregó py-spy como dependencia del proyecto.

Se detectó que el calentamiento construía y usaba el detector en el hilo de startup, pero las validaciones posteriores lo usaban en hilos AnyIO/asyncio diferentes. La muestra identifica la espera en MediaPipe; atribuirla exclusivamente a esa migración requiere la comparación posterior, no solo la observación de una pila.

Corrección: el worker conserva un ejecutor de un solo hilo para creación, calentamiento y procesamiento. La llamada HTTP permanece fuera del loop asíncrono durante el cálculo; SQL continúa en su capa/event loop. FaceMesh se reutiliza por hilo mediante almacenamiento local, evitando que otra instancia de worker herede un grafo creado por un hilo distinto. `app/main.py` cierra el ejecutor y limpia la instancia en shutdown. Los contratos y reglas de los retos se mantienen.

Se agregaron pruebas para comprobar que startup y peticiones desde varios hilos usan el mismo propietario de inferencia, y que dos hilos distintos no comparten el objeto FaceMesh. La aceptación de rendimiento sigue pendiente del ensayo real de esta versión y de la prueba con cámara; los resultados anteriores no se presentan como mejora lograda.

Regresión de esta versión con hilo propietario y cierre ordenado: **136 aprobadas, 2 omitidas, 46 advertencias**; lint y `git diff --check` aprobados. Las advertencias incluyen las deprecaciones de los eventos startup/shutdown existentes de FastAPI y las de Pydantic; no son resultados de reconocimiento facial. Las dos omisiones conservan los motivos descritos arriba. El log actualizado es `ci-evidence/hu46-cache-regression.log`.

### Medición real de la versión con hilo propietario

La imagen local se reconstruyó y se inició con `init`, conservando el volumen de modelos y PostgreSQL. El servicio completó startup. Ensayo `hu46-owner-thread`, primera ronda, con catorce JPEG diferentes de la misma fotografía pública:

| Fase HTTP del ensayo | Frames | Duración | Resultado |
|---|---:|---:|---|
| Primer grupo nuevo | 7 | 15,4445 s | HTTP 200, captura estática rechazada |
| Segundo grupo nuevo | 7 | 9,0685 s | HTTP 200, captura estática rechazada |
| Revalidación de los catorce frames | 14 | 0,0722 s | HTTP 200, captura estática rechazada |

Suma de esas peticiones: **24,5852 segundos**. Evidencia: `ci-evidence/hu46-owner-thread-http-flow.log`. La revalidación pudo reutilizar las inferencias y volvió a aplicar las reglas, conservando el rechazo de una foto inmóvil. Esto demuestra respuesta del micro y reducción del trabajo repetido en ese ensayo; no acredita la precisión con personas ni el tiempo total de asistencia.

La prueba con siete frames nuevos de la versión anterior a la propiedad del hilo había superado el timeout de 180 segundos, registrado en `ci-evidence/hu46-cache-http-flow-final.log`. Las condiciones y la carga del equipo variaron; no se calcula un porcentaje general de mejora ni se equipara el reporte manual de 215 segundos con esta suma. El arranque en frío también siguió siendo largo y se documenta como limitación; no se incluye en los tiempos de peticiones con el servicio preparado.

**Estado actual:** servicio local operativo y ensayo HTTP aprobado; la velocidad aún no se declara aceptada para una ficha de cuarenta personas. Persisten entre 9 y 15 segundos por grupo nuevo en esta observación. QA debe medir dos asistencias reales consecutivas con el servicio preparado, separando tiempo de captura, retos, guardado y respuesta de la interfaz. Si la meta exige menos tiempo, debe acordarse un objetivo medible y evaluar el costo restante de inferencia sin reducir comprobaciones de identidad o presencia real. No hay commit/push ni despliegue en producción de estos cambios.

Comprobaciones adicionales reales: `/health` respondió `ok` desde Windows en `127.0.0.1:8091`; la fotografía pública completa con varias personas devolvió HTTP 400 y `La imagen contiene mas de un rostro.`. No se inscribió ninguna persona ni se insertó asistencia durante estos ensayos. Evidencia de rechazo: `ci-evidence/hu46-owner-thread-multiple-faces.log`. La base habitual se mantuvo en su contenedor, sin reinicios ni cambios de esquema por esta corrección.
