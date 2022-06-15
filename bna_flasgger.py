# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:33:01 2021

@author: charan
"""

import pandas as pd
import pickle
from flask import Flask, request
from flasgger import Swagger

app = Flask(__name__)
bna_model_pkl = open("saved_models/bn_classifier.pkl" , 'rb')
classifier = pickle.load(bna_model_pkl)
Swagger(app)

@app.route("/predict", methods =["GET"])
def predict():
    """ authenticating single note data
    This is using flasgger api
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
        
    responses:
        200:
            description: the predicted output of the given single note data
    """
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "the prediction for the given values are " + str(prediction[0])

@app.route("/predict_file", methods = ["POST"])
def predict_file():
    """ authenticating the data of notes given in the file
    This is using flasgger api
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: the predicted output of the notes data given in the file
    """
    data = pd.read_csv(request.files.get("file"))
    predictions = classifier.predict(data)
    return "the predictions for the data given in file is "+ str(list(predictions))

if __name__ == "__main__":
    app.run()