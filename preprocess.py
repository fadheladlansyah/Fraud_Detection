import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import pickle
import json
import gc



class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr):
        self.attributes = attr

    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.attributes].values


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder_ = []

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            le = LabelEncoder().fit(X[:,col])
            self.label_encoder_.append(le)
        return self

    def transform(self, X):
        for col in range(X.shape[1]):
            X[:,col] = self.label_encoder_[col].transform(X[:,col])
        return X

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def map_email(df, mapping):
    for col in ['P_emaildomain', 'R_emaildomain']:
        df[col] = df[col].map(mapping)

    return df        


def map_DeviceInfo(df, col='DeviceInfo'):
    df['have_DeviceInfo'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('unknown').str.lower()
    df[col] = df[col].str.split('/', expand=True)[0]
    df[col] = df[col].apply(lambda x: x.lower())

    df.loc[df[col].str.contains('windows', na=False), col] = 'Windows'
    df.loc[df[col].str.contains('sm-', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('samsung', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('gt-', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('moto', na=False), col] = 'Motorola'
    df.loc[df[col].str.contains('lg-', na=False), col] = 'LG'
    df.loc[df[col].str.contains('rv:', na=False), col] = 'RV'
    df.loc[df[col].str.contains('huawei', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('ale-', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('-l', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('hi6', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('blade', na=False), col] = 'ZTE'
    df.loc[df[col].str.contains('trident', na=False), col] = 'Trident'
    df.loc[df[col].str.contains('lenovo', na=False), col] = 'Lenovo'
    df.loc[df[col].str.contains('xt', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('f3', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('f5', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('lt', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('htc', na=False), col] = 'HTC'
    df.loc[df[col].str.contains('mi', na=False), col] = 'Xiaomi'

    cat_so_far = ['unknown', 'windows', 'ios device', 'macos', 'Samsung', 'Trident', 'RV',
                    'Motorola', 'Huawei', 'LG', 'Sony', 'ZTE', 'HTC', 'Lenovo', 'Xiaomi']
    df[col] = df[col].apply(lambda x: x if x in cat_so_far else 'other')

    return df


def map_id30(df, col='id_30'):
    df['have_id30'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('unknown')
    df[col] = df[col].str.split(' ', expand=True)[0]
    
    return df


def map_id31(df, col='id_31'):
    df['have_id31'] = df[col].isna().astype(int)
    df[col] = df[col].fillna('unknown')

    df.loc[df[col].str.contains('chrome', na=False), col] = 'chrome'
    df.loc[df[col].str.contains('safari', na=False), col] = 'safari'
    df.loc[df[col].str.contains('firefox', na=False), col] = 'firefox'
    df.loc[df[col].str.contains('edge', na=False), col] = 'edge'
    df.loc[df[col].str.contains('ie', na=False), col] = 'ie'
    df.loc[df[col].str.contains('android', na=False), col] = 'default'
    df.loc[df[col].str.contains('samsung', na=False), col] = 'default'
    df.loc[df[col].str.contains('browser', na=False), col] = 'default'

    df.loc[df[col].isin(df[col].value_counts()[df[col].value_counts() < 200].index), col] = "other"

    return df


def map_id33(df, col='id_33'):
    df['have_id33'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('0')
    df[col] = df[col].str.split('x', expand=True)[0].astype(int)
    df[col] = df[col].map(lambda x: int(x/1000))

    return df


def preprocess(df, preprocess_attr, pca_pipeline, imputer_encoder_pipeline):
    # preprocess_attr = json.load(open('model/preprocess_attr.json', 'r'))
    # pca_pipeline = pickle.load(open('model/pca_pipeline.pkl', 'rb'))
    # imputer_encoder_pipeline = pickle.load(open('model/imputer_encoder_pipeline.pkl', 'rb'))
    
    df.drop(['isFraud'], axis=1, inplace=True)

    df.drop(preprocess_attr['cols_to_drop'], axis=1, inplace=True)
    df = map_email(df, preprocess_attr['emails_mapping'])
    df = map_DeviceInfo(df, col='DeviceInfo')
    df = map_id30(df, col='id_30')
    df = map_id31(df, col='id_31')
    df = map_id33(df, col='id_33')

    temp = pca_pipeline.transform(df)
    temp = pd.DataFrame(temp, index=df.index)
    temp.rename(columns=lambda x: 'V_pc_'+str(x), inplace=True)

    df = pd.concat([df.drop(preprocess_attr['applied_pca_features'], axis=1), temp],
                   axis=1)

    num_features = preprocess_attr['numerical_features']
    cat_features = preprocess_attr['categorical_features']
    temp = imputer_encoder_pipeline.transform(df)
    df = pd.DataFrame(data=temp, columns=num_features+cat_features)

    gc.collect()

    return df



# df_samples = pd.read_csv('dataset/df_samples.csv')
# random_ = np.random.randint(0,200)
# instance_ = pd.DataFrame(df_samples.iloc[random_]).T
# instance_ = preprocess(instance_)

# print(instance_.shape)
# print(instance_['TransactionAmt'])
# print(instance_['card4'])
# print(instance_['card6'])
    