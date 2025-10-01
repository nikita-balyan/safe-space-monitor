# gunicorn.conf.py
import os

# Gunicorn config for Render
bind = "0.0.0.0:" + str(int(os.environ.get("PORT", 10000)))
workers = 1  # Use only 1 worker on Render free tier
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increase timeout to 2 minutes
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True  # Preload app to save memory

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Prevent worker timeouts
graceful_timeout = 120