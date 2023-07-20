import click
import os
import logging
from flask.cli import FlaskGroup, with_appcontext, pass_script_info
from .main import init_app

logger = logging.getLogger(__name__)

