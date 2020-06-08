from flask import Flask
app = Flask(__name__)
from flask import render_template, jsonify, redirect, url_for, request

import numpy as np
import pandas as pd

import joblib
import json
import pickle
import sys
sys.path.append('./dataset/')
sys.path.append('./model/')

from preprocess import preprocess, DataFrameSelector, MultiColumnLabelEncoder


@app.route('/')


@app.route('/home')
def home():
    return render_template('home.html', title='Home')


@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        random_ = np.random.randint(0,200)
        instance_ = df_samples.iloc[random_:random_+1,:]

        instance_['TransactionAmt'] = np.float64(request.form['TransactionAmt'])
        instance_['card4'] = str(request.form['card4'])
        instance_['card6'] = str(request.form['card6'])
        
        instance_preprocessed_ = preprocess(instance_, preprocess_attr, pca_pipeline, imputer_encoder_pipeline)

        isFraud = clf.predict(instance_preprocessed_)[0]

        return render_template('result.html',
                                instance_=list(instance_.T.to_dict().values())[0],
                                isFraud=isFraud,
                                )
                    
    if request.method == 'GET':
        return render_template('predict.html', title='Predict')


if __name__ == '__main__':
    preprocess_attr = json.load(open('model/preprocess_attr.json', 'r'))
    pca_pipeline = pickle.load(open('model/pca_pipeline.pkl', 'rb'))
    imputer_encoder_pipeline = pickle.load(open('model/imputer_encoder_pipeline.pkl', 'rb'))

    df_samples = pd.read_csv('dataset/df_samples.csv')
    clf = joblib.load('model/gridCV_log.joblib') 

    app.run(debug=True)