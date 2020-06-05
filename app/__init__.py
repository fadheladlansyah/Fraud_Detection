from flask import Flask
app = Flask(__name__)

from flask import render_template, jsonify, redirect, url_for, request

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression

import joblib

import pickle
import json
import itertools
import random
import gc
import sys
sys.path.append('../dataset/')
sys.path.append('../model/')
import warnings
warnings.filterwarnings("ignore")

import preprocess


df_samples = pd.read_csv('dataset/df_samples.csv')
preprocess_attr = json.load(open('preprocess_attr.json', 'rb'))
pca_pipeline = pickle.load(open('pca_pipeline.pkl', 'rb'))
imputer_pipeline = pickle.load(open('imputer_pipeline.pkl', 'rb'))
categorical_encoder = pickle.load(open('categorical_encoder.pkl', 'rb'))
clf = joblib.load('model/gridCV_log.joblib')

df_samples = preprocess.map_email(df_samples, preprocess_attr['emails_mapping'])
df_samples = preprocess.map_DeviceInfo(df_samples, col='DeviceInfo')
df_samples = preprocess.map_id30(df_samples, col='id_30')
df_samples = preprocess.map_id31(df_samples, col='id_31')
df_samples = preprocess.map_id33(df_samples, col='id_33')
df_samples = pca_pipeline.transform(df_samples)
df_samples = imputer_pipeline.transform(df_samples)


@app.route('/')

@app.route('/home')
def home():
    return render_template('home.html', title='Home')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        features = ['TransactionAmt', 'card4', 'card6']

        input_ = df_samples.sample()

        for feat in features:
            input_[feat] = request.form[feat]
        

        df_samples[features].iloc[0].values.reshape(1,-1)



        return render_template('result.html', title='Prediction Result')
                    
    if request.method == 'GET':
        return render_template('predict.html', title='Predict')


# @app.route('/result')