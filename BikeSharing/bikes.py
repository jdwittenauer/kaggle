# -*- coding: utf-8 -*-
"""
@author: John Wittenauer

@notes: This script was tested on 64-bit Windows 7 using the Anaconda 2.0
distribution of 64-bit Python 2.7.
"""

import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import *
from sklearn.ensemble import *
from sklearn.grid_search import *
from sklearn.feature_selection import *
from sklearn.learning_curve import *


def performance_test():
    """
    Test NumPy performance.  Should run in less than a second on most machines.
    """
    A = np.random.random((2000, 2000))
    B = np.random.random((2000, 2000))
    t = time.time()
    np.dot(A, B)
    print(time.time()-t)


def load_model(filename):
    """
    Load a previously training model from disk.
    """
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model


def save_model(model, filename):
    """
    Persist a trained model to disk.
    """
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def generate_features(data):
    """
    Generates new derived features to add to the data set for model training.
    """
    print('TODO')

    return data


def process_training_data(filename, ex_generate_features):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv(filename, sep=',')
    num_features = len(training_data.columns) - 3

    # drop the total count label and move the registered/casual counts to the front
    cols = training_data.columns.tolist()
    cols = cols[:1] + cols[-3:-1] + cols[1:num_features]
    training_data = training_data[cols]

    if ex_generate_features:
        training_data = generate_features(training_data)

    num_features = len(training_data.columns)
    X = training_data.iloc[:, 3:num_features].values
    y1 = training_data.iloc[:, 1].values
    y2 = training_data.iloc[:, 2].values

    return training_data, X, y1, y2


def main():
    ex_process_training_data = True
    ex_generate_features = False
    ex_load_model = False
    ex_save_model = False

    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\BikeSharing\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\BikeSharing\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'

    algorithm = 'bayes'  # bayes, logistic, svm, sgd, forest, boost
    metric = None  # accuracy, f1, rcc_auc, mean_absolute_error, mean_squared_error, r2_score

    training_data = None
    X = None
    y1 = None
    y2 = None
    model = None

    os.chdir(code_dir)

    print('Starting process...')

    if ex_process_training_data:
        print('Reading in training data...')
        training_data, X, y1, y2 = process_training_data(data_dir + training_file, ex_generate_features)

    if ex_load_model:
        print('Loading model from disk...')
        model = load_model(data_dir + model_file)

    if ex_save_model:
        print('Saving model to disk...')
        save_model(model, data_dir + model_file)

    print('Process complete.')


if __name__ == "__main__":
    main()