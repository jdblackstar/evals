FROM python:3.11-slim
COPY packages/auth /srv/auth
COPY packages/shared /srv/shared
WORKDIR /srv
CMD ["uvicorn", "auth.service:app"]
