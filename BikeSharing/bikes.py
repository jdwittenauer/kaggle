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


def performance_test(x):
    """
    Test NumPy performance.  Use to compare computation speed across machines.
    """
    A = np.random.random((x, x))
    B = np.random.random((x, x))
    t = time.time()
    np.dot(A, B)
    print(time.time() - t)


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


def create_transforms(X, transform_list, transforms):
    """
    Creates transform objects to apply before training or scoring.
    """
    for k in transform_list:
        if k == 'scaler':
            # create a standardization transform
            transforms[k] = preprocessing.StandardScaler()
            transforms[k].fit(X)
        elif k == 'pca':
            # create a PCA transform
            transforms[k] = decomposition.PCA(whiten=True)
            transforms[k].fit(X)
        elif k == 'selector':
            # create a feature selection transform
            transforms[k] = VarianceThreshold(threshold=0.0)
            transforms[k].fit(X)

    return transforms


def apply_transforms(X, transforms):
    """
    Applies pre-computed transformations to a data set.
    """
    for k in transforms:
        if transforms[k] is not None:
            X = transforms[k].transform(X)

    return X


def visualize_feature_distributions(training_data, X, y1, y2, viz_type, max_features):
    """
    Generates feature distribution plots (histogram or kde) for each feature.
    """
    print('TODO')


def visualize_correlations(training_data, X, y1, y2, viz_type, max_features):
    """
    Generates a correlation matrix heat map.
    """
    print('TODO')


def visualize_pairwise_relationships(training_data, X, y1, y2, viz_type, max_features):
    """
    Generates a faucet plot showing pairwise relationships between factors.
    """
    print('TODO')


def visualize_sequential_relationships(training_data, X, y1, y2, viz_type, max_features):
    """
    Generates line plots to visualize sequential data.
    """
    print('TODO')


def visualize_principal_components(training_data, X, y1, y2, viz_type, max_features):
    """
    Generates scatter plots to visualize the principal components of the data set.
    """
    print('TODO')


def define_model(model_type, algorithm):
    """
    Defines and returns a model object of the designated type.
    """
    model = None

    if model_type == 'classification':
        if algorithm == 'bayes':
            model = naive_bayes.GaussianNB()
        elif algorithm == 'logistic':
            model = linear_model.LogisticRegression(penalty='l2', C=1.0)
        elif algorithm == 'svm':
            model = svm.SVC(C=1.0, kernel='rbf', shrinking=True, probability=False, cache_size=200)
        elif algorithm == 'sgd':
            model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001,
                                               n_iter=1000, shuffle=False, n_jobs=-1)
        elif algorithm == 'forest':
            model = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto', max_depth=None,
                                           min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'boost':
            model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features=None,
                                               max_leaf_nodes=None)
        else:
            print('No model defined for ' + algorithm)
            exit()
    else:
        print('TODO')

    return model


def train_model(X, y, transforms, model_type, algorithm):
    """
    Trains a new model using the training data.
    """
    t0 = time.time()
    model = define_model(model_type, algorithm)
    X = apply_transforms(X, transforms)
    model.fit(X, y)
    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    return model


def visualize_feature_importance(training_data, model, offset):
    """
    Generates a feature importance plot.  Requires a trained random forest or gradient boosting model.
    """
    print('Generating feature importance plot...')

    importance = model.feature_importances_
    importance = 100.0 * (importance / importance.max())
    importance = importance[0:30]
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0])

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Variable Importance')
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(training_data.columns[sorted_idx + offset])
    ax.set_xlabel('Relative Importance')

    fig.tight_layout()


def predict(X, model, transforms):
    """
    Predicts the class label.
    """
    X = apply_transforms(X, transforms)
    y_est = model.predict(X)

    return y_est


def predict_probability(X, model, transforms):
    """
    Predicts the class probabilities.
    """
    X = apply_transforms(X, transforms)
    y_prob = model.predict_proba(X)[:, 1]

    return y_prob


def score(X, y, model, transforms):
    """
    Scores the model's performance and returns the result.
    """
    X = apply_transforms(X, transforms)

    return model.score(X, y)


def cross_validate(X, y, model_type, algorithm, metric, transforms):
    """
    Performs cross-validation to estimate the true performance of the model.
    """
    model = define_model(model_type, algorithm)
    X = apply_transforms(X, transforms)

    t0 = time.time()
    scores = cross_validation.cross_val_score(model, X, y, scoring=metric, cv=3, n_jobs=-1)
    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def main():
    ex_process_training_data = True
    ex_generate_features = False
    ex_create_transforms = False
    ex_load_model = False
    ex_save_model = False
    ex_visualize_feature_distributions = False
    ex_visualize_correlations = False
    ex_visualize_pairwise_relationships = False
    ex_visualize_sequential_relationships = False
    ex_visualize_principal_components = False

    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\BikeSharing\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\BikeSharing\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'

    algorithm = 'bayes'  # bayes, logistic, svm, sgd, forest, boost
    metric = None  # accuracy, f1, rcc_auc, mean_absolute_error, mean_squared_error, r2_score
    transform_list = ['scaler', 'pca', 'selector']
    transforms = {'scaler': None, 'pca': None, 'selector': None}

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

    if ex_create_transforms:
        transforms = create_transforms(X, transform_list, transforms)

    if ex_load_model:
        print('Loading model from disk...')
        model = load_model(data_dir + model_file)

    if ex_save_model:
        print('Saving model to disk...')
        save_model(model, data_dir + model_file)

    if ex_visualize_feature_distributions:
        print('Visualizing feature distributions...')
        visualize_feature_distributions(training_data, X, y1, y2, None, None)

    if ex_visualize_correlations:
        print('Visualizing feature distributions...')
        visualize_correlations(training_data, X, y1, y2, None, None)

    if ex_visualize_pairwise_relationships:
        print('Visualizing feature distributions...')
        visualize_pairwise_relationships(training_data, X, y1, y2, None, None)

    if ex_visualize_sequential_relationships:
        print('Visualizing feature distributions...')
        visualize_sequential_relationships(training_data, X, y1, y2, None, None)

    if ex_visualize_principal_components:
        print('Visualizing feature distributions...')
        visualize_principal_components(training_data, X, y1, y2, None, None)

    print('Process complete.')


if __name__ == "__main__":
    main()