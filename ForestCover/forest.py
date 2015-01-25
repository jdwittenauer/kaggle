# -*- coding: utf-8 -*-
"""
@author: John Wittenauer

@notes: This script was tested on 64-bit Windows 7 using the Anaconda 2.0
distribution of 64-bit Python 2.7.
"""

import os
import math
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import *
from sklearn.ensemble import *


def load(filename):
    """
    Load a previously training model from disk.
    """
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model


def save(model, filename):
    """
    Persist a trained model to disk.
    """
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def process_training_data(filename, features, standardize, whiten):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv(filename, sep=',')

    X = training_data.iloc[:, 0:features].values
    y = training_data.iloc[:, features+1].values

    # create a standardization transform
    scaler = None
    if standardize:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)

    # create a PCA transform
    pca = None
    if whiten:
        pca = decomposition.PCA(whiten=True)
        pca.fit(X)

    return training_data, X, y, scaler, pca


def visualize(training_data, X, y, scaler, pca, features):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.

    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """

    # feature histograms
    num_histograms = features / 16 if features % 16 == 0 else features / 16 + 1
    for i in range(num_histograms):
        fig, ax = plt.subplots(4, 4, figsize=(20, 10))
        for j in range(16):
            index = (i * 16) + j
            if index < features:
                ax[j % 4, j / 4].hist(X[:, index], bins=30)
                ax[j % 4, j / 4].set_title(training_data.columns[index + 1])
                ax[j % 4, j / 4].set_xlim((min(X[:, index]), max(X[:, index])))
        fig.tight_layout()

    # correlation matrix
    if scaler is not None:
        X = scaler.transform(X)

    fig2, ax2 = plt.subplots(figsize=(16, 10))
    colormap = sb.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF", "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
    sb.corrplot(training_data, annot=False, sig_stars=False, diag_names=False, cmap=colormap, ax=ax2)
    fig2.tight_layout()

    # pca plots
    if pca is not None:
        X = pca.transform(X)

        fig4, ax4 = plt.subplots(figsize=(16, 10))
        ax4.scatter(X[:, 0], X[:, 1], c=y)
        ax4.set_title('First & Second Principal Components')

        fig5, ax5 = plt.subplots(figsize=(16, 10))
        ax5.scatter(X[:, 1], X[:, 2], c=y)
        ax5.set_title('Second & Third Principal Components')


def train(X, y, algorithm, scaler, pca, fit):
    """
    Trains a new model using the training data.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if algorithm == 'bayes':
        model = naive_bayes.GaussianNB()
    elif algorithm == 'logistic':
        model = linear_model.LogisticRegression()
    elif algorithm == 'svm':
        model = svm.SVC()
    elif algorithm == 'forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=200,
                                       min_samples_leaf=200, max_features=30)
    elif algorithm == 'boost':
        model = GradientBoostingClassifier(n_estimators=100, max_depth=7, min_samples_split=200,
                                           min_samples_leaf=200, max_features=30)
    else:
        print 'No model defined for ' + algorithm
        exit()

    if fit:
        t0 = time.time()
        model.fit(X, y)
        t1 = time.time()
        print 'Model trained in {0:3f} s.'.format(t1 - t0)

        if algorithm == 'forest' or algorithm == 'boost':
            # generate a plot showing model performance and relative feature importance
            importances = model.feature_importances_

    return model


def predict(X, model, scaler, pca):
    """
    Predicts the class label.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    y_est = model.predict(X)

    return y_est


def predict_probability(X, model, scaler, pca):
    """
    Predicts the class probabilities.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    y_prob = model.predict_proba(X)[:, 1]

    return y_prob


def score(X, y, model, scaler, pca, metric):
    """
    Create weighted signal and background sets and calculate the AMS.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if metric is None:
        y_est = model.score(X, y)

    return y_est


def cross_validate(X, y, algorithm, scaler, pca):
    """
    Performs cross-validation to estimate the true performance of hte model.
    """
    return None


def process_test_data():
    """
    Reads in the test data set and prepares it for prediction by the model.
    """
    return None


def generate_submission_file():
    """
    Create a new submission file with test data and predictions generated by the model.
    """
    return None


def generate_features():
    """
    Generates new derived features to add to the data set for model training.
    """
    return None


def select_features():
    """
    Selects a subset of the total number of features available.
    """
    return None


def ensemble():
    """
    Creates an ensemble of many models together.
    """
    return None


def main():
    # perform some initialization
    load_training_data = True
    load_model = False
    train_model = False
    save_model = False
    create_visualizations = True
    create_submission_file = False
    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\ForestCover\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\ForestCover\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'
    features = 54
    algorithm = 'logistic'  # logistic, bayes, svm, forest, boost
    standardize = False
    whiten = False

    os.chdir(code_dir)

    print 'Starting process...'
    print 'algorithm={0}, standardize={1}, whiten={2}'.format(algorithm, standardize, whiten)

    if load_training_data:
        print 'Reading in training data...'
        training_data, X, y, scaler, pca = process_training_data(
            data_dir + training_file, features, standardize, whiten)

    if create_visualizations:
        print 'Creating visualizations...'
        visualize(training_data, X, y, scaler, pca, features)

    if load_model:
        print 'Loading model from disk...'
        model = load(data_dir + model_file)

    if train_model:
        print 'Training model on full data set...'
        model = train(X, y, algorithm, scaler, pca, False)

        print 'Calculating predictions...'
        # TODO

        print 'Calculating score...'
        # TODO

        print 'Performing cross-validation...'
        # TODO

    if save_model:
        print 'Saving model to disk...'
        # TODO

    if create_submission_file:
        print 'Reading in test data...'
        # TODO

        print 'Predicting test data...'
        # TODO

        print 'Creating submission file...'
        # TODO

    print 'Process complete.'


if __name__ == "__main__":
    main()