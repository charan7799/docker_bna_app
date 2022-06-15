# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:28:09 2021

@author: charan
"""

import pandas as pd
import pickle
from flask import Flask, request

app = Flask(__name__)
bna_model_pkl = open("saved_models/bn_classifier.pkl" , 'rb')
classifier = pickle.load(bna_model_pkl)

@app.route("/")
def Welcome():
    return "Welcome!!"

@app.route("/predict")
def predict():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "the prediction for the given values are " + str(prediction[0])

@app.route("/predict_file", methods = ["POST"])
def predict_file():
    data = pd.read_csv(request.files.get("file"))
    predictions = classifier.predict(data)
    return "the predictions for the data given in file is "+ str(list(predictions))

if __name__ == "__main__":
    app.run()