from flask import Flask

app = Flask("segmentation-server")

# Late-initialized in run.py
processor = None

from server.app import routes
