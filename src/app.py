from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, render_template, request, redirect, url_for, abort
from PIL import Image, ImageOps

import _init_paths
import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory



app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)


    if request.method == "GET":
        return render_template("upload.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/flask_input.jpg"
        f.save(filepath)
        opt.demo = filepath
        # # 画像ファイルを読み込む
        image = [opt.demo]
        # image = Image.open(file[0])

        # 予測を実施
        detector.run(image[0])

        return render_template("upload_done.html", path="./static/flask_output.jpg")


if __name__ == "__main__":
    opt = opts().init()

    if os.path.exists("./static/flask_output.jpg"):
        os.remove("./static/flask_output.jpg")
    if os.path.exists("./static/flask_input.jpg"):
        os.remove("./static/flask_input.jpg")

    app.run(debug=True)