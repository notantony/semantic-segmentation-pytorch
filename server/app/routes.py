import base64
import os
import json
import tempfile
import server.app
import traceback
import flask
import io

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
            extension = json_data.get("type", "")
            image_path = os.path.join(tmpdir, "image.{}".format(extension))
            with open(image_path, "wb") as image_file:
                image_file.write(data)
            name = json_data["name"]
            blur_radius = json_data.get("blur_radius", None)
            border_extension = json_data.get("border_extension", None)
        except Exception as e:
            traceback.print_exc()
            return bad_request(repr(e))

        if name not in server.app.processor.from_names:
            return bad_request("Unexpected name: `{}`".format(name))

        try:
            result_image = server.app.processor.get_segment(image_path, name, blur_radius, border_extension)
        except Exception as e:
            traceback.print_exc()
            return server_error(repr(e))
        
        bbox = result_image.getbbox()
        if bbox is None:
            return bad_request("Image doesn't contain class `{}`".format(name))
        result_image = result_image.crop(bbox)
        img_stream = io.BytesIO()
        result_image.save(img_stream, format="png")

        response = {}
        response["image"] = base64.encodebytes(img_stream.getvalue()).decode("utf-8").replace("\n", "")
        
        return jsonify(response)


@app.route('/colormap', methods=['POST'])
def colormap():
    with tempfile.TemporaryDirectory("crop-session") as tmpdir:
        if request.mimetype == "application/json":
            try:
                json_data = json.loads(request.get_data())
                image_data = base64.decodebytes(json_data["data"].encode())
                extension = json_data.get("type", "")
            except Exception as e:
                traceback.print_exc()
                return bad_request(repr(e))
        elif request.mimetype == "image/jpeg" or request.mimetype == "image/png":
            image_data = request.get_data()
        else:
            return bad_request("Unsupported MIME type, use `application/json`")

        image_path = os.path.join(tmpdir, "image.{}".format(extension))
        with open(image_path, "wb") as image_file:
            image_file.write(image_data)

        try:
            colormap = server.app.processor.get_colormap(image_path)
        except Exception as e:
            traceback.print_exc()
            return server_error(repr(e))
        
        response = {}
        response["colormap"] = base64.b64encode(colormap.tobytes()).decode("utf-8").replace("\n", "")
        response["shape"] = str(colormap.shape)
        response["dtype"] = str(colormap.dtype)
        
        return jsonify(response)
