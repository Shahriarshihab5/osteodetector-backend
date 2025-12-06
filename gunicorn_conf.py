# gunicorn_conf.py
import multiprocessing

workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5 minutes
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
preload_app = True  # Load model once, share across workers
bind = "0.0.0.0:10000"
