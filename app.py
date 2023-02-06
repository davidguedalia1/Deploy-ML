import pickle
import sys
from flask import render_template, request, Flask
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocess import Preprocess

application = app = Flask(__name__)

MODEL_PATH = 'model.pkl'
file = open(MODEL_PATH, 'rb')
model = pickle.load(file)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        print('This is error output', file=sys.stderr)
        print('This is standard output', file=sys.stdout)
        msg = request.form['message']
        prediction = preprocess_sample(msg)
        output = prediction[0]
        if output == 1:
            my_prediction = "Fake News"
        else:
            my_prediction = "Satire"
        res = render_template('result.html', res=my_prediction)
        return res

   
def preprocess_sample(msg):
    list_msg = msg.splitlines()
    lst = []
    body = ""
    count = 0
    lst = ["", "", ""]
    for line in list_msg:
        if count == 0:
            if len(line.strip()) > 5:
                lst[0] = line.strip()
            else:
                lst[0] = ("no_title")
        elif count == 1:
            if "http" in line or "www" in line:
                lst[1] = line.strip()
            else:
                lst[1] = ("no_url")
        elif count > 1:
            body += line.strip()
        count += 1
    lst[2] = body
    sample = pd.DataFrame([lst], columns=['title', 'url', 'body'])
    to_predict = Preprocess(sample).create_data_to_predict()
    prediction = model.predict(to_predict)
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)