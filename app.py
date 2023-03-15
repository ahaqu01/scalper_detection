# coding=utf-8
from concurrent.futures import ThreadPoolExecutor
import json
import os
import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from gevent import pywsgi
import sys
# import torch
# from torch.multiprocessing import Process
from threading import Thread

sys.path.append("./src")
from scalper_demo import scalper_demo

# for logging
import time
import logging
from logging.handlers import TimedRotatingFileHandler  # 设置日志按时间回滚
from functools import wraps

# torch.multiprocessing.set_start_method('spawn')

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, supports_credentials=True)


# 使用多线程进行异步调用
def asyncc(f):
    wraps(f)

    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


def log(logfolder):
    if not os.path.exists(logfolder):
        os.makedirs(logfolder)
    # 初始化logging
    logging.basicConfig()
    logger = logging.getLogger()
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 添加TimeRoatingFileHandler
    # 定义一个1天换一次log文件的handler
    # 保留7个旧log文件
    timefilehandler = logging.handlers.TimedRotatingFileHandler(os.path.join(logfolder, "log.log"), when='D',
                                                                interval=1, backupCount=7, encoding="utf-8", )
    timefilehandler.suffix = "%Y-%m-%d.log"
    # 设置log记录输出的格式
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(filename)s-%(lineno)d-%(message)s')
    timefilehandler.setFormatter(formatter)
    # 添加到logger中
    logger.addHandler(timefilehandler)


@asyncc
def person_feature(jsonobj):
    output_results = {}

    try:
        url = jsonobj["url"]
        visual = jsonobj["visual"]
        video_root = jsonobj["video_root"]
        callback = jsonobj["callback"]
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
        output_results["error_code"] = "500"
        output_results["error_msg"] = str(e)
        logging.info("-----one segment fail-----")

    # 回传结果
    try:
        res = requests.post(callback, json=output_results)
        logging.info("callback success!")
    except:
        logging.info("callback fail!")



@asyncc
def other_function():
    """
    其他异步调用函数
    """
    pass


# 接收到请求，立刻返回并异步调用黄牛检测算法
@app.route("/scalper_det", methods=['POST'])
def res():
    result = {
        "status_code": 0,
        "err_message": ""
    }
    try:
        data = request.get_data(as_text=True)
        logging.info("get data: {}".format(data))
        result["status_code"] = 200

    except Exception as e:
        logging.error("get data error: {}".format(str(e)))
        result["status_code"] = 500
        result["err_message"] = "get data error: {}".format(str(e))

    else:

        # 异步调用
        scalper_det_main(data)

    response = make_response(jsonify(result), 200)

    return response


@asyncc
def scalper_det_main(data):
    jsonobj = json.loads(data)
    try:
        person_feature(jsonobj)
    except Exception as e:
        logging.error("-----send callback failed with error:" + str(e))


if __name__ == '__main__':
    import sys

    log_folder = sys.argv[1]
    port = sys.argv[2]
    log(log_folder)
    sd = scalper_demo()
    print("******* 服务启动********")
    server = pywsgi.WSGIServer(('0.0.0.0', int(port)), app)
    print('Listening on address: 0.0.0.0:{}'.format(port))
    server.serve_forever()
