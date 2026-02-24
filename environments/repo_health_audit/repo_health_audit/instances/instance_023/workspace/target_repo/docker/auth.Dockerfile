FROM python:3.11-slim
COPY services/auth /srv/auth
COPY services/common /srv/common
WORKDIR /srv
EXPOSE 8001
CMD ["uvicorn", "auth.handler:app", "--host", "0.0.0.0", "--port", "8001"]
