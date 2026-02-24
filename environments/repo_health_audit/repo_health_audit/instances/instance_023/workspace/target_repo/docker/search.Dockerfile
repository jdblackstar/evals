FROM python:3.11-slim
COPY services/search /srv/search
COPY services/common /srv/common
WORKDIR /srv
EXPOSE 8002
CMD ["uvicorn", "search.query:app", "--host", "0.0.0.0", "--port", "8002"]
