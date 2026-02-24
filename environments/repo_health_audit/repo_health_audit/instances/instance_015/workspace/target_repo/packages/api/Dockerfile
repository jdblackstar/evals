FROM python:3.11-slim

WORKDIR /app

COPY packages/shared /app/packages/shared
COPY packages/api /app/packages/api
COPY pyproject.toml /app/

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "packages.api.src.server:app", "--host", "0.0.0.0", "--port", "8000"]
