FROM python:3.11-slim
COPY services/worker /worker
CMD ["python", "/worker/runner.py"]
