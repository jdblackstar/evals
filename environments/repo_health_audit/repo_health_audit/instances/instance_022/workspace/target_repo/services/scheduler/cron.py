from celery import Celery
from celery.schedules import crontab

app = Celery("scheduler", broker="redis://localhost:6379/0")

app.conf.beat_schedule = {
    "etl-hourly": {
        "task": "services.scheduler.cron.run_etl",
        "schedule": crontab(minute=0),
    },
}


@app.task
def run_etl():
    # Placeholder — would invoke ETL pipeline
    return {"status": "completed"}


@app.task
def run_cleanup():
    return {"status": "cleaned"}
