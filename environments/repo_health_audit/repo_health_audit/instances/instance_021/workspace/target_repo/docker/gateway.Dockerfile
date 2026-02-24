FROM python:3.11-slim
COPY packages/gateway /srv/gateway
COPY packages/shared /srv/shared
WORKDIR /srv
EXPOSE 8000
CMD ["uvicorn", "gateway.router:app", "--host", "0.0.0.0"]
