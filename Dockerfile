FROM python:3.11-slim AS build

ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=10
ENV PATH="/opt/venv/bin:$PATH"
ENV HOME=/home/facelit
ENV ENVIRONMENT=production

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 facelit \
    && useradd --uid 10001 --gid 10001 --create-home facelit \
    && mkdir -p /home/facelit/.insightface /app/.cache/matplotlib \
    && chown -R 10001:10001 /home/facelit /app/.cache

COPY --from=build /opt/venv /opt/venv

COPY app ./app

EXPOSE 8090
USER 10001:10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=3).close()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
