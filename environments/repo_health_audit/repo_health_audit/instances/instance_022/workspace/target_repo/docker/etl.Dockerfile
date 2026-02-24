FROM python:3.11-slim
RUN pip install pandas numpy sqlalchemy psycopg2-binary boto3 pyarrow
COPY services/etl /srv/etl
COPY lib /srv/lib
WORKDIR /srv
CMD ["python", "-m", "etl.extract"]
