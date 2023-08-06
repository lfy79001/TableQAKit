import os


working_dir = os.path.dirname(os.path.abspath(__file__))
workers = 1
bind = "210.75.240.136:18888"
accesslog = "gunicorn_access.log"
errorlog = "gunicorn_error.log"
loglevel = "info"
timeout = 300
