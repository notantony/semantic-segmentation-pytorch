import base64
import os
import json
import tempfile
import server.app
import traceback
import flask

from flask import request, jsonify
from server.app import app
from server.app.utils import bad_request, server_error
from config import cfg


@app.route('/')
@app.route('/index')
def index():
    return "Index"


@app.route('/segmentation', methods=['POST'])
def segmentation():
    with tempfile.TemporaryDirectory("crop-session") as tmpdir:
        if request.mimetype != "application/json":
            return bad_request("Unsupported MIME type, use `application/json`")
        try:
            json_data = json.loads(request.get_data())
            data = base64.decodebytes(json_data["data"].encode())
            extension = json_data["type"]
            image_path = os.path.join(tmpdir, "image.{}".format(extension))
            with open(image_path, "wb") as image_file:
                image_file.write(data)
            name = json_data["name"]
        except Exception as e:
            traceback.print_exc()
            return bad_request(repr(e))

        try: # TODO: Unsupported class
            result_image = server.app.processor.get_segment(image_path, name)
        except Exception as e:
            traceback.print_exc()
            return server_error(repr(e))
        
        result_image = result_image.crop(result_image.getbbox()) # TODO: Missing class
        result_path = os.path.join(tmpdir, "result.png")
        result_image.save(result_path)
        
        response = {}
        with open(result_path, "rb") as result_file:
            result_data = result_file.read()
        response["image"] = str(base64.encodebytes(result_data))
        
        return jsonify(response)
