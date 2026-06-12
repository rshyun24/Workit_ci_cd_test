import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('workit')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Redis 3.x 호환
app.conf.broker_transport_options = {
    'visibility_timeout': 3600,
    'socket_connect_timeout': 10,
}