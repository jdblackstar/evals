FROM python:3.11-slim
COPY services/ingest /srv/ingest
COPY services/common /srv/common
WORKDIR /srv
CMD ["python", "-m", "ingest.parser"]
