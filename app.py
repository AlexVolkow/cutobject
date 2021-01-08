import json
import os

import tensorflow as tf
from flask import Flask, flash, request, redirect, send_file, send_from_directory
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename

from cutobject import CutObject
from segmentation import Segmentation

UPLOAD_FOLDER = 'tmp'

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
segmentation = Segmentation()
cutter = CutObject(segmentation, UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder='node_modules')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(filename)


@app.route('/', methods=['GET', 'POST'])
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
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            crop_params = request.files['crop'].read()
            crop_json = json.loads(crop_params)
            crop = (int(crop_json['y']), int(crop_json['x']),
                    int(crop_json['y'] + crop_json['height']), int(crop_json['x'] + crop_json['width']))

            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                output_path = cutter.cut(image_path, crop)

            return send_file(output_path)
    return send_from_directory("ui", "index.html")


if __name__ == '__main__':
    app.run(debug=True)
