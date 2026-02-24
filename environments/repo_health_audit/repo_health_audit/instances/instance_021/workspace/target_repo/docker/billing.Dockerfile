FROM python:3.11-slim
COPY packages/billing /srv/billing
COPY packages/shared /srv/shared
WORKDIR /srv
CMD ["uvicorn", "billing.invoice:app"]
