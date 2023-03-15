# coding=utf-8
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent import pywsgi

import sys
sys.path.append("./src")
from scalper_demo import scalper_demo

# for logging
import time
import logging
from logging.handlers import TimedRotatingFileHandler  # 设置日志按时间回滚

if not os.path.exists("./logs"):
    os.makedirs("./logs")

# 初始化logging
logging.basicConfig()
logger = logging.getLogger()
# 设置日志级别
logger.setLevel(logging.INFO)
# 添加TimeRoatingFileHandler
# 定义一个1天换一次log文件的handler
# 保留7个旧log文件
timefilehandler = logging.handlers.TimedRotatingFileHandler("./logs/log.log", when='D',
                                                            interval=1, backupCount=7, encoding="utf-8", )
timefilehandler.suffix = "%Y-%m-%d.log"
# 设置log记录输出的格式
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(filename)s-%(lineno)d-%(message)s')
timefilehandler.setFormatter(formatter)
# 添加到logger中
logger.addHandler(timefilehandler)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, supports_credentials=True)


@app.route("/scalper_det", methods=['POST'])
def scalper_det():
    output_results = {}
    try:
        data = request.get_data(as_text=True)
        logging.info("get data: {}".format(data))
        jsonObj = json.loads(data)
        url = jsonObj["url"]
        visual = jsonObj["visual"]
        video_root = jsonObj["video_root"]
        output_results_demo = sd.demo(url, visual=visual, video_root=video_root)
        output_results.update(output_results_demo)
        # produce results for jsonify
        output_results["conclusion"] = str(output_results["conclusion"])
        video_segment_results = output_results["video_segment_results"]
        for i, item in enumerate(video_segment_results):
            video_segment_results[i].pop("face_embedding")
            video_segment_results[i].pop("person_feature")
            video_segment_results[i]["huangniu_confidence"] = str(video_segment_results[i]["huangniu_confidence"])
        logging.info("-----finish one segment-----")
    except Exception as e:
        logging.error("error: {}".format(str(e)))
        output_results["error_code"] = "400"
        output_results["error_msg"] = str(e)
        logging.info("-----one segment fail-----")
    output_results = jsonify(output_results)
    return output_results


if __name__ == '__main__':
    sd = scalper_demo()
    print("******* 服务启动********")
    # app.run(debug=True, host="192.168.1.44", port=6677)
    server = pywsgi.WSGIServer(('0.0.0.0', 6677), app)
    print('Listening on address: 0.0.0.0:6677')
    server.serve_forever()
