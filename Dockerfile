FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 --retries=10 -r requirements.txt

COPY app ./app

# Preload InsightFace so the first enrollment request does not spend minutes
# downloading and preparing the model.
RUN python -c "from app.workers.face_embedding_worker import get_face_embedding_worker; get_face_embedding_worker().warm_up()"

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
