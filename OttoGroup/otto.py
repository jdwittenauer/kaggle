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
from keras.utils import np_utils


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
    y_prob = model.predict_proba(X)

    return y_prob


def score(X, y, algorithm, model, scaler):
    X = apply_scaler(X, scaler)

    if algorithm == 'xgb':
        y_est = model.predict(X)
        return log_loss(y, y_est)
    else:
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


def define_model(algorithm, num_features, num_classes):
    model = None

    if algorithm == 'logistic':
        model = LogisticRegression(penalty='l2', C=1.0)
    elif algorithm == 'xgb':
        params = {'target': 'target',
                  'max_iterations': 250,
                  'max_depth': 10,
                  'min_child_weight': 4,
                  'row_subsample': .9,
                  'min_loss_reduction': 1,
                  'column_subsample': .8}
        model = xgb.XGBClassifier(params)
    elif algorithm == 'nn':
        model = Sequential()
        model.add(Dense(num_features, 512, init='glorot_uniform'))
        model.add(PReLU((512,)))
        model.add(BatchNormalization((512,)))
        model.add(Dropout(0.5))

        model.add(Dense(512, 512, init='glorot_uniform'))
        model.add(PReLU((512,)))
        model.add(BatchNormalization((512,)))
        model.add(Dropout(0.5))

        model.add(Dense(512, 512, init='glorot_uniform'))
        model.add(PReLU((512,)))
        model.add(BatchNormalization((512,)))
        model.add(Dropout(0.5))

        model.add(Dense(512, num_classes, init='glorot_uniform'))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train_model(X, y, y_onehot, algorithm, model, scaler):
    t0 = time.time()
    X = apply_scaler(X, scaler)

    if algorithm == 'logistic' or algorithm == 'xgb':
        model.fit(X, y)
    elif algorithm == 'nn':
        model.fit(X, y_onehot, nb_epoch=20, batch_size=16, validation_split=0.15)

    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    return model


def cross_validate(X, y, algorithm, scaler, folds=3):
    model = define_model(algorithm)
    X = apply_scaler(X, scaler)
    t0 = time.time()

    if algorithm == 'xgb':
        scores = []
        kf = KFold(y.shape[0], n_folds=3, shuffle=True)
        for train_index, test_index in kf:
            model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
            predictions = model.predict(X[test_index])
            actuals = y[test_index]
            scores.append(log_loss(actuals, predictions))
    else:
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
    model_file = 'keras.pkl'
    algorithm = 'nn'

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
    model = define_model(algorithm, num_features, num_classes)

    print('Training model...')
    model = train_model(X, y, y_onehot, algorithm, model, scaler)

    if algorithm == 'logistic' or algorithm == 'xgb':
        print('Training score = ' + str(score(X, y, algorithm, model, scaler)))

        print('Running cross-validation...')
        val_score = cross_validate(X, y, algorithm, scaler)
        print('Cross-validation score = ' + str(val_score))

    print ('Saving model...')
    save_model(model, data_dir + model_file)

    print('Generating submission file...')
    y_prob = predict_probability(X_test, model, scaler)
    make_submission(y_prob, ids, encoder, data_dir, submit_file)

    print('Script complete.')


if __name__ == "__main__":
    main()
