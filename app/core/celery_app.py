from celery import Celery
import os

# Specify the Redis address (service name from docker-compose)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create a Celery instance
celery_app = Celery(
    "pipesim_worker",
    broker=REDIS_URL,
    backend=REDIS_URL # The results of the tasks will be stored here
)

# Settings for stable video operation
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Moscow",
    enable_utc=True,
)