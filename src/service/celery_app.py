# celery_app.py
from celery import Celery
from kombu import Exchange, Queue
from src.config import settings

BROKER_HOST = settings.RABBITMQ_HOST
BROKER_VHOST = settings.RABBITMQ_VHOST
BROKER_PORT = settings.RABBITMQ_PORT
BROKER_USER = settings.RABBITMQ_USERNAME
BROKER_PASSWORD = settings.RABBITMQ_PASSWORD

# Redis configuration
REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_PASSWORD = settings.REDIS_PASSWORD

redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

app = Celery(
    'ocr_service',
    broker=f"amqp://{BROKER_USER}:{BROKER_PASSWORD}@{BROKER_HOST}:{BROKER_PORT}/{BROKER_VHOST}",
    backend=redis_url
)

default_exchange = Exchange('celery_tasks', type='direct', durable=True)

app.conf.imports = ['src.service.celery_task']
app.conf.worker_max_tasks_per_child = 50

app.conf.task_queues = (
    Queue('process_file', default_exchange, routing_key='process_file', queue_arguments={'x-queue-type': 'quorum'}),
    Queue('ocr_file', default_exchange, routing_key='ocr_file', queue_arguments={'x-queue-type': 'quorum'}),
    Queue('combine_results', default_exchange, routing_key='combine_results', queue_arguments={'x-queue-type': 'quorum'}),
)

app.conf.task_routes = {
    'task.process_file': {'queue': 'process_file', 'routing_key': 'process_file'},
    'task.ocr_file': {'queue': 'ocr_file', 'routing_key': 'ocr_file'},
    'task.combine_results': {'queue': 'combine_results', 'routing_key': 'combine_results'},
}

app.conf.broker_transport_options = {
    "confirm_publish": True,
}