# -*- coding: utf-8 -*-
"""
@author: John Wittenauer

@notes: This script was tested on Linux using the Anaconda 2.2
distribution of 32-bit Python 2.7.
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from sklearn.cross_validation import *
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.ensemble import *

import xgboost as xgb

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils


def load_model(filename):
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    return model


def save_model(model, filename):
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def predict(X, model, scaler):
    X = apply_scaler(X, scaler)
    y_est = model.predict(X)

    return y_est


def predict_probability(X, model, scaler):
    X = apply_scaler(X, scaler)
    y_prob = model.predict_proba(X)[:, 1]

    return y_prob


def score(X, y, model, scaler):
    X = apply_scaler(X, scaler)

    return model.score(X, y)


def load_training_data(path, filename):
    df = pd.read_csv(path + filename)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    return X, labels


def load_test_data(path, filename):
    df = pd.read_csv(path + filename)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    return X, ids


def create_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def apply_scaler(X, scaler):
    X = scaler.transform(X)
    return X


def preprocess_labels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    y_onehot = np_utils.to_categorical(y)
    return y, y_onehot, encoder


def define_model(algorithm):
    model = None

    if algorithm == 'logistic':
        model = LogisticRegression(penalty='l2', C=1.0)

    return model


def train_model(X, y, model, scaler):
    t0 = time.time()
    X = apply_scaler(X, scaler)
    model.fit(X, y)
    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    return model


def cross_validate(X, y, algorithm, scaler, folds=3):
    model = define_model(algorithm)
    X = apply_scaler(X, scaler)

    t0 = time.time()
    scores = cross_val_score(model, X, y, cv=folds, n_jobs=-1)
    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def make_submission(y_prob, ids, encoder, path, filename):
    with open(path + filename, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probabilities in zip(ids, y_prob):
            p = ','.join([i] + [str(p) for p in probabilities.tolist()])
            f.write(p)
            f.write('\n')


def main():
    code_dir = '/home/john/git/kaggle/OttoGroup/'
    data_dir = '/home/john/data/otto/'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    algorithm = 'logistic'

    os.chdir(code_dir)
    np.random.seed(1337)

    print('Starting script...')

    print('Loading data...')
    X, labels = load_training_data(data_dir, training_file)
    X_test, ids = load_test_data(data_dir, test_file)

    print('Pre-processing...')
    scaler = create_scaler(X)
    y, y_onehot, encoder = preprocess_labels(labels)
    num_features = X.shape[1]
    num_classes = y_onehot.shape[1]
    print('Features = ' + str(num_features))
    print('Classes = ' + str(num_classes))

    print('Building model...')
    model = define_model(algorithm)

    print('Training model...')
    model = train_model(X, y, model, scaler)
    print('Training score = ' + str(score(X, y, model, scaler)))

    print('Running cross-validation...')
    val_score = cross_validate(X, y, algorithm, scaler)
    print('Cross-validation score = ' + str(val_score))

    print ('Saving model...')
    save_model(model, data_dir + 'model.pkl')

    print('Generating submission file...')
    y_prob = model.predict_proba(X_test)
    make_submission(y_prob, ids, encoder, data_dir, submit_file)

    print('Script complete.')


if __name__ == "__main__":
    main()
