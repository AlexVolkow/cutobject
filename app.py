import json
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, flash, request, redirect, send_from_directory, send_file
from tensorflow.python.keras.backend import set_session

from cutobject import CutObject
from segmentation import Segmentation

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
segmentation = Segmentation()
cutter = CutObject(segmentation)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder='node_modules')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/demo/cutout', methods=['GET', 'POST'])
def cutout():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            crop_params = request.files['crop'].read()
            crop_json = json.loads(crop_params)
            crop = (int(crop_json['y']), int(crop_json['x']),
                    int(crop_json['y'] + crop_json['height']), int(crop_json['x'] + crop_json['width']))

            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                output = cutter.cut(image, crop)

            res = BytesIO()
            output.save(res, "png")
            res.seek(0)

            return send_file(res,
                             mimetype='image/png',
                             attachment_filename="foreground.png")

    return send_from_directory("ui", "index.html")


if __name__ == '__main__':
    app.run(debug=True)
