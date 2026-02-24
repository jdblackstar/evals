FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
COPY services/notifications/requirements.txt /app/services/notifications/
RUN pip install --no-cache-dir -r requirements.txt -r services/notifications/requirements.txt

COPY libs/ /app/libs/
COPY services/notifications/ /app/services/notifications/

CMD ["celery", "-A", "services.notifications.sender", "worker", "--loglevel=info"]
