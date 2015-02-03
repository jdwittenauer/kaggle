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
from sklearn.learning_curve import *
from sklearn.grid_search import *


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


def process_training_data(filename, features, impute):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv(filename, sep=',')

    # impute missing values
    if impute == 'mean':
        training_data.fillna(training_data.mean())
    elif impute == 'zeros':
        training_data.fillna(0)

    X = training_data.iloc[:, 1:features].values
    y = training_data.iloc[:, features+1].values

    return training_data, X, y


def create_transforms(X, standardize, whiten):
    """
    Creates transform objects to apply before training or scoring.
    """
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

    return scaler, pca


def apply_transforms(X, scaler, pca):
    """
    Applies pre-computed transformations to a data set.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    return X


def visualize(training_data, X, y, scaler, pca, features):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.

    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """

    print('Generating individual feature histograms...')
    num_histograms = features / 16 if features % 16 == 0 else features / 16 + 1
    for i in range(num_histograms):
        fig, ax = plt.subplots(4, 4, figsize=(20, 10))
        for j in range(16):
            index = (i * 16) + j
            if index < (features - 1):
                ax[j / 4, j % 4].hist(X[:, index], bins=30)
                ax[j / 4, j % 4].set_title(training_data.columns[index + 1])
                ax[j / 4, j % 4].set_xlim((min(X[:, index]), max(X[:, index])))
            elif index < features:
                ax[j / 4, j % 4].hist(y, bins=30)
                ax[j / 4, j % 4].set_title(training_data.columns[index + 1])
                ax[j / 4, j % 4].set_xlim((min(y), max(y)))
        fig.tight_layout()

    print('Generating correlation matrix...')
    if scaler is not None:
        X = scaler.transform(X)

    fig2, ax2 = plt.subplots(figsize=(16, 10))
    colormap = sb.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF", "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
    sb.corrplot(training_data, annot=False, sig_stars=False, diag_names=False, cmap=colormap, ax=ax2)
    fig2.tight_layout()

    if pca is not None:
        print('Generating principal component plots...')
        X = pca.transform(X)
        class_count = np.count_nonzero(np.unique(y))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        fig3, ax3 = plt.subplots(figsize=(16, 10), projection='3d')
        for i in range(class_count):
            class_idx = i + 1  # add 1 if class index start at 1 instead of 0
            ax3.scatter(X[y == class_idx, 0], X[y == class_idx, 1], X[y == class_idx, 2],  c=colors[i], label=class_idx)
        ax3.set_title('First & Second Principal Components')
        ax3.legend()
        fig3.tight_layout()

        fig4, ax4 = plt.subplots(figsize=(16, 10))
        for i in range(class_count):
            class_idx = i + 1  # add 1 if class index start at 1 instead of 0
            ax4.scatter(X[y == class_idx, 1], X[y == class_idx, 2], c=colors[i], label=class_idx)
        ax4.set_title('Second & Third Principal Components')
        ax4.legend()
        fig4.tight_layout()

        fig5, ax5 = plt.subplots(figsize=(16, 10))
        for i in range(class_count):
            class_idx = i + 1  # add 1 if class index start at 1 instead of 0
            ax5.scatter(X[y == class_idx, 2], X[y == class_idx, 3], c=colors[i], label=class_idx)
        ax5.set_title('Third & Fourth Principal Components')
        ax5.legend()
        fig5.tight_layout()


def define_model(algorithm):
    """
    Defines and returns a model object of the designated type.
    """
    model = None

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

    return model


def train(training_data, X, y, algorithm, scaler, pca):
    """
    Trains a new model using the training data.
    """
    t0 = time.time()
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca)
    model.fit(X, y)
    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    if algorithm == 'forest' or algorithm == 'boost':
        print('Generating feature importance plot...')
        fig, ax = plt.subplots(figsize=(16, 10))

        importance = model.feature_importances_
        importance = 100.0 * (importance / importance.max())
        importance = importance[0:30]
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0])
        ax.set_title('Variable Importance')
        ax.barh(pos, importance[sorted_idx], align='center')
        ax.set_yticks(pos)
        ax.set_yticklabels(training_data.columns[sorted_idx + 1])
        ax.set_xlabel('Relative Importance')

        fig.tight_layout()

    return model


def predict(X, model, scaler, pca):
    """
    Predicts the class label.
    """
    X = apply_transforms(X, scaler, pca)
    y_est = model.predict(X)

    return y_est


def predict_probability(X, model, scaler, pca):
    """
    Predicts the class probabilities.
    """
    X = apply_transforms(X, scaler, pca)
    y_prob = model.predict_proba(X)[:, 1]

    return y_prob


def score(X, y, model, scaler, pca):
    """
    Create weighted signal and background sets and calculate the AMS.
    """
    X = apply_transforms(X, scaler, pca)

    return model.score(X, y)


def cross_validate(X, y, algorithm, scaler, pca, metric):
    """
    Performs cross-validation to estimate the true performance of the model.
    """
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca)

    t0 = time.time()
    scores = cross_validation.cross_val_score(model, X, y, scoring=metric, cv=3, n_jobs=-1, verbose=1)
    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def plot_learning_curve(X, y, algorithm, scaler, pca, metric):
    """
    Plots a learning curve showing model performance against both training and
    validation data sets as a function of the number of training samples.
    """

    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca)

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring=metric, cv=3, n_jobs=-1, verbose=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                    alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.legend(loc='best')
    fig.tight_layout()


def parameter_search(X, y, algorithm, scaler, pca, metric):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca)
    param_grid = None

    if algorithm == 'bayes':
        param_grid = None
    elif algorithm == 'logistic':
        param_grid = None
    elif algorithm == 'svm':
        param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    elif algorithm == 'sgd':
        param_grid = None
    elif algorithm == 'forest':
        param_grid = None
    elif algorithm == 'boost':
        param_grid = None

    grid_estimator = GridSearchCV(model, param_grid, scoring=metric, cv=3, n_jobs=-1, verbose=1)
    grid_estimator.fit(X, y)

    return grid_estimator.best_estimator_, grid_estimator.best_params_, grid_estimator.best_score_


def process_test_data(filename, features, impute):
    """
    Reads in the test data set and prepares it for prediction by the model.
    """
    test_data = pd.read_csv(filename, sep=',')

    # impute missing values
    if impute == 'mean':
        test_data.fillna(test_data.mean())
    elif impute == 'zeros':
        test_data.fillna(0)

    X_test = test_data.iloc[:, 1:features].values

    return test_data, X_test


def create_submission(test_data, y_est, submit_file):
    """
    Create a new submission file with test data and predictions generated by the model.
    """
    submit = pd.DataFrame(columns=['Id', 'Cover_Type'])
    submit['Id'] = test_data['Id']
    submit['Cover_Type'] = y_est
    submit.to_csv(submit_file, sep=',', index=False, index_label=False)


def plot_roc_curve():
    """
    Plots a receiver operating characteristic (ROC) curve for the given classifier.
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
    create_visualizations = True
    load_model = False
    train_model = False
    create_learning_curve = False
    perform_grid_search = False
    save_model = False
    create_submission_file = False
    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\ForestCover\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\ForestCover\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'
    features = 54
    algorithm = 'logistic'  # bayes, logistic, svm, sgd, forest, boost
    impute = 'none'  # zeros, mean, none
    metric = None  # accuracy, f1, rcc_auc, mean_absolute_error, mean_squared_error, r2_score
    standardize = False
    whiten = False

    os.chdir(code_dir)

    print('Starting process...')
    print('Algorithm={0}, Impute={1} Standardize={2}, Whiten={3}'.format(algorithm, impute, standardize, whiten))
    training_data, X, y, scaler, pca, model = None

    if load_training_data:
        print('Reading in training data...')
        training_data, X, y = process_training_data(data_dir + training_file, features, impute)

    if standardize or whiten:
        print('Creating data transforms...')
        scaler, pca = create_transforms(X, standardize, whiten)

    if create_visualizations:
        print('Creating visualizations...')
        visualize(training_data, X, y, scaler, pca, features)

    if load_model:
        print('Loading model from disk...')
        model = load(data_dir + model_file)

    if train_model:
        print('Training model on full data set...')
        model = train(training_data, X, y, algorithm, scaler, pca)

        print('Calculating training score...')
        model_score = score(X, y, model, scaler, pca)
        print('Training score ='), model_score

        if create_learning_curve:
            print('Generating learning curve...')
            plot_learning_curve(X, y, algorithm, scaler, pca, metric)
        else:
            print('Performing cross-validation...')
            cross_val_score = cross_validate(X, y, algorithm, scaler, pca, metric)
            print('Cross-validation score ='), cross_val_score

    if perform_grid_search:
        print('Performing hyper-parameter grid search...')
        best_model, best_params, best_score = parameter_search(X, y, algorithm, scaler, pca, metric)
        print('Best model = ', best_model)
        print('Best params = ', best_params)
        print('Best score = ', best_score)

    if save_model:
        print('Saving model to disk...')
        save(model, data_dir + model_file)

    if create_submission_file:
        print('Reading in test data...')
        test_data, X_test = process_test_data(data_dir + test_file, features, impute)

        print('Predicting test data...')
        y_est = predict(X_test, model, scaler, pca)

        print('Creating submission file...')
        create_submission(test_data, y_est, data_dir + submit_file)

    print('Process complete.')


if __name__ == "__main__":
    main()