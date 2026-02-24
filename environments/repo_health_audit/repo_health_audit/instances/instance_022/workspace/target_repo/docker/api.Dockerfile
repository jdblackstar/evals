FROM python:3.11-slim
RUN pip install fastapi uvicorn pydantic sqlalchemy psycopg2-binary
COPY services/api /srv/api
COPY lib /srv/lib
WORKDIR /srv
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0"]
