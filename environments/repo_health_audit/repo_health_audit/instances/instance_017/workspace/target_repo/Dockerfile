FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e .

COPY pipelines/ /app/pipelines/
COPY shared/ /app/shared/
COPY noxfile.py /app/

CMD ["python", "-m", "pipelines.ingest"]
