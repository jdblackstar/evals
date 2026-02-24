FROM python:3.11-slim

WORKDIR /app

COPY packages/shared /app/packages/shared
COPY packages/worker /app/packages/worker
COPY pyproject.toml /app/

RUN pip install --no-cache-dir -e .

CMD ["celery", "-A", "packages.worker.src.tasks", "worker", "--loglevel=info"]
