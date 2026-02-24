FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
COPY services/users/requirements.txt /app/services/users/
RUN pip install --no-cache-dir -r requirements.txt -r services/users/requirements.txt

COPY libs/ /app/libs/
COPY services/users/ /app/services/users/

EXPOSE 5001
CMD ["python", "-m", "flask", "--app", "services.users.api", "run", "--host=0.0.0.0", "--port=5001"]
