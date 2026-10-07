from flask import Flask, request, jsonify




import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 
import os
import datetime
import string
from flask_cors import CORS

home_path = r"C:\Users\hp\Desktop\Final OCR\HasnatOCR"  # Change only this path 


##   Adding Paths shange according to your device 

import sys
sys.path.insert(0, os.path.join(home_path,'src','data'))
import preproc as pp, evaluation
from generator import DataGenerator, Tokenizer
from reader import Dataset
 
sys.path.insert(0, os.path.join(home_path,'src','network'))
from model import HTRModel

##########################################################################




##   defining required paths and  variables




charfile=open(os.path.join(home_path, "chars (1).txt"),"r",encoding='utf-8').read()

# define parameters
source = "urdu_data"
arch = "bluche"


# define paths
source_path = os.path.join(home_path, "data", f"{source}.hdf5")
output_path = os.path.join(home_path, "output", source, arch)
target_path = os.path.join(output_path, "checkpoint_weights.hdf5")


# define input size, number max of chars per line and list of valid chars
input_size = (1024, 128, 1)
max_text_length = 128
# charset_base = string.printable[:95]
# charset_base=urdu_chracters()
charset_base=charfile


##################################################################################



##  Loading models and tokenizer

tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/recognition', methods=['POST','GET'])
def recognition():
    
    image = request.json
    image = np.array(image, dtype=np.uint8)
        # Create an empty 3-channel image of the same size as the binary image
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Copy the binary image to all three channels of the RGB image
    rgb_image[:, :, 0] = image
    rgb_image[:, :, 1] = image
    rgb_image[:, :, 2] = image


    image = rgb_image.copy()
    img = pp.preprocess_predict(image, input_size=input_size)
    x_test = pp.normalization([img])

    model1 = HTRModel(architecture=arch,
                    input_size=input_size,
                    vocab_size=tokenizer.vocab_size,
                    beam_width=10,
                    top_paths=10)
    model1.compile(learning_rate=0.001)
    model1.load_checkpoint(target=target_path)
    predicts, probabilities = model1.predict(x_test, ctc_decode=True)
    predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

    for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
        max_prob_idx = np.argmax(prob)
        max_prob = prob[max_prob_idx]
        max_pred = pred[max_prob_idx][::-1]
        
        
    return jsonify(max_pred)


if __name__ == '__main__':
    app.run(debug=True,port = 5004)



    