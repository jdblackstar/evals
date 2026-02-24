FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
COPY services/gateway/requirements.txt /app/services/gateway/
RUN pip install --no-cache-dir -r requirements.txt -r services/gateway/requirements.txt

COPY libs/ /app/libs/
COPY services/gateway/ /app/services/gateway/

EXPOSE 5000
CMD ["python", "-m", "flask", "--app", "services.gateway.proxy", "run", "--host=0.0.0.0", "--port=5000"]
