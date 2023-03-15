import os
import re
import time

import requests
import json
import base64
import numpy as np
import cv2
from gevent import pywsgi
from threading import Thread

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, supports_credentials=True)

import sys
port = sys.argv[1]
callback_port = sys.argv[2]

requests_url = "http://127.0.0.1:{}/scalper_det".format(port)
headers = {
    'Connection': 'close',
}


def base64_to_numpy(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


@app.route("/callback", methods=["POST"])
def recv_data():
    data = request.get_data(as_text=True)
    print(data)
    return ""


def demo(video_url):
    input_json = {
        "url": video_url,
        "visual": True,
        "video_root": "/workspace/huangniu_det/visual",
        "callback": "http://127.0.0.1:{}/callback".format(callback_port)
    }
    response = requests.post(requests_url, headers=headers, json=input_json)
    print(response)
    print("Status code:", response.status_code)
    res = json.loads(response.text)
    print(res)
    # for item in res["video_segment_results"]:
    #     global_id = item["global_person_id"]
    #     if "face_img" in item and item["face_img"] is not None:
    #         face_base_64 = item["face_img"]
    #         face = base64_to_numpy(face_base_64)
    #         cv2.imwrite("{}_face.jpg".format(global_id), face)
    #     if "full_body_image" in item and item["full_body_image"] is not None:
    #         full_body_base_64 = item["full_body_image"]
    #         full_body = base64_to_numpy(full_body_base_64)
    #         cv2.imwrite("{}_body.jpg".format(global_id), full_body)


def run():
    server = pywsgi.WSGIServer(('0.0.0.0', int(callback_port)), app)
    print('Listening on address: 0.0.0.0:{}'.format(callback_port))
    server.serve_forever()


t = Thread(target=run)
t.start()
demo("/workspace/huangniu_det/ed40384e00b942ce04da37bd73208082.mp4")
print("send")
