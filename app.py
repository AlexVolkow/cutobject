import os

import tensorflow as tf
from flask import Flask, flash, request, redirect, send_file
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename

from matting import PyMatting
from segmentation import Segmentation

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
segmentation = Segmentation()
matting = PyMatting(segmentation)

UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
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

            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                output_path = matting.matting(image_path, UPLOAD_FOLDER)

            return send_file(output_path)
    return '''
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
