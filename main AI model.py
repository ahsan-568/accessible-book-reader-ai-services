from flask_cors import CORS
from flask import Flask, request, jsonify

import pypdfium2 as pdfium
import numpy as np


app = Flask(__name__)

CORS(app)

app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['pdf'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
    return 'Homepage'

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    # check if the post request has the file part
    if 'files' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['files']

    # errors = {}
    success = True
 
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        pdf = pdfium.PdfDocument(file.read())
        n_pages = len(pdf)
        images_list=list()
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render_topil(scale=1,rotation=0,crop=(0, 0, 0, 0),greyscale=False,optimise_mode=pdfium.OptimiseMode.NONE,)
            images_list.append(np.array(pil_image).astype('uint8'))
        im=dict()
        count=1
        for i in images_list:
            js = i.tolist()
            im[str(count)]=js
            count+=1
        response = jsonify(im)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

app.run(debug=True, port=5001)











