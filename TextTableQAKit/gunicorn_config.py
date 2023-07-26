import os

working_dir = os.path.dirname(os.path.abspath(__file__))
workers = 4
bind = "210.75.240.136:18890"
accesslog = "gunicorn_access.log"
errorlog = "gunicorn_error.log"
loglevel = "info"