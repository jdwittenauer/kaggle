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
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.grid_search import *
from sklearn.learning_curve import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.preprocessing import *
from sklearn.svm import *


def performance_test(x):
    """
    Test NumPy performance.  Use to compare computation speed across machines.
    """
    A = np.random.random((x, x))
    B = np.random.random((x, x))
    t = time.time()
    np.dot(A, B)
    print(time.time() - t)


def load_csv_data(directory, filename, dtype=None, index=None, coerce_index=False):
    """
    Test NumPy performance.  Use to compare computation speed across machines.
    """
    data = pd.read_csv(directory + filename, sep=',', dtype=dtype)

    if index is not None and coerce_index:
        if type(index) is str:
            data[index] = data[index].convert_objects(convert_dates='coerce')
        else:
            for key in index:
                data[key] = data[key].convert_objects(convert_dates='coerce')

        data = data.set_index(index)

    print('Data file ' + filename + ' loaded successfully.')

    return data


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


def generate_features(data):
    """
    Generates new derived features to add to the data set for model training.
    """
    print('TODO')

    return data


def process_training_data(directory, filename, ex_generate_features):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = load_csv_data(directory, filename)
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


def create_transforms(X, transform_list, transforms, missing='NaN', impute_strategy='mean', categories=None):
    """
    Creates transform objects to apply before training or scoring.
    """
    for k in transform_list:
        if k == 'imputer':
            # impute missing values
            transforms[k] = Imputer(missing_values=missing, strategy=impute_strategy)
            transforms[k].fit(X)
        elif k == 'onehot':
            # create a category encoder
            transforms[k] = OneHotEncoder(categorical_features=categories, sparse=False)
            transforms[k].fit(X)
        elif k == 'scaler':
            # create a standardization transform
            transforms[k] = StandardScaler()
            transforms[k].fit(X)
        elif k == 'pca':
            # create a PCA transform
            transforms[k] = PCA(whiten=True)
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


def visualize_regression_residuals(training_data, X, y1, y2, viz_type, max_features):
    """
    Generates line plots to visualize residuals from a regression function.
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
            model = GaussianNB()
        elif algorithm == 'logistic':
            model = LogisticRegression(penalty='l2', C=1.0)
        elif algorithm == 'svm':
            model = SVC(C=1.0, kernel='rbf', shrinking=True, probability=False, cache_size=200)
        elif algorithm == 'sgd':
            model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, n_iter=1000, shuffle=False, n_jobs=-1)
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
        if algorithm == 'ridge':
            model = Ridge(alpha=1.0)
        elif algorithm == 'svm':
            model = SVR(C=1.0, kernel='rbf', shrinking=True, probability=False, cache_size=200)
        elif algorithm == 'sgd':
            model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, n_iter=1000, shuffle=False)
        elif algorithm == 'forest':
            model = RandomForestRegressor(n_estimators=10, criterion='mse', max_features='auto', max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'boost':
            model = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                              min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features=None,
                                              max_leaf_nodes=None)
        else:
            print('No model defined for ' + algorithm)
            exit()

    return model


def train_model(X, y, model_type, algorithm, transforms):
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


def visualize_feature_importance(training_data, model, column_offset):
    """
    Generates a feature importance plot.  Requires a trained random forest or gradient boosting model.
    """
    importance = model.feature_importances_
    importance = 100.0 * (importance / importance.max())
    importance = importance[0:30]
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0])

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Variable Importance')
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(training_data.columns[sorted_idx + column_offset])
    ax.set_xlabel('Relative Importance')

    fig.tight_layout()


def cross_validate(X, y, model_type, algorithm, metric, transforms):
    """
    Performs cross-validation to estimate the true performance of the model.
    """
    model = define_model(model_type, algorithm)
    X = apply_transforms(X, transforms)

    t0 = time.time()
    scores = cross_val_score(model, X, y, scoring=metric, cv=3, n_jobs=-1)
    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def plot_learning_curve(X, y, model_type, algorithm, metric, transforms):
    """
    Plots a learning curve showing model performance against both training and
    validation data sets as a function of the number of training samples.
    """
    model = define_model(model_type, algorithm)
    X = apply_transforms(X, transforms)

    t0 = time.time()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring=metric, cv=3, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    alpha=0.1, color='r')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                    alpha=0.1, color='r')
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation score')
    ax.legend(loc='best')
    fig.tight_layout()
    t1 = time.time()
    print('Learning curve generated in {0:3f} s.'.format(t1 - t0))


def parameter_search(X, y, model_type, algorithm, metric, transforms):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    model = define_model(model_type, algorithm)
    X = apply_transforms(X, transforms)

    param_grid = None
    if algorithm == 'logistic':
        param_grid = [{'penalty': ['l1', 'l2'], 'C': [0.1, 0.3, 1.0, 3.0]}]
    elif algorithm == 'ridge':
        param_grid = [{'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]}]
    elif algorithm == 'svm':
        param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    elif algorithm == 'sgd':
        param_grid = [{'loss': ['hinge', 'log', 'modified_huber'], 'penalty': ['l1', 'l2'],
                       'alpha': [0.0001, 0.001, 0.01], 'iter': [100, 1000, 10000]}]
    elif algorithm == 'forest':
        param_grid = [{'n_estimators': [10, 30, 100, 300], 'criterion': ['gini', 'entropy'],
                       'max_features': ['auto', 'log2', None], 'max_depth': [3, 5, 7, None],
                       'min_samples_split': [2, 10, 30, 100], 'min_samples_leaf': [1, 3, 10, 30, 100]}]
    elif algorithm == 'boost':
        param_grid = [{'learning_rate': [0.1, 0.3, 1.0], 'subsample': [1.0, 0.9, 0.7, 0.5],
                       'n_estimators': [100, 300, 1000], 'max_features': ['auto', 'log2', None],
                       'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 10, 30, 100],
                       'min_samples_leaf': [1, 3, 10, 30, 100]}]

    t0 = time.time()
    grid_estimator = GridSearchCV(model, param_grid, scoring=metric, cv=3, n_jobs=-1)
    grid_estimator.fit(X, y)
    t1 = time.time()
    print('Grid search completed in {0:3f} s.'.format(t1 - t0))

    return grid_estimator.best_estimator_, grid_estimator.best_params_, grid_estimator.best_score_


def train_ensemble(X, y, model_type, algorithm, transforms):
    """
    Creates an ensemble of many models together.
    """
    model = define_model(model_type, algorithm)
    X = apply_transforms(X, transforms)

    t0 = time.time()
    ensemble_model = BaggingClassifier(base_estimator=model, n_estimators=10, max_samples=1.0, max_features=1.0,
                                       bootstrap=True, bootstrap_features=False)
    ensemble_model.fit(X, y)
    t1 = time.time()
    print('Ensemble training completed in {0:3f} s.'.format(t1 - t0))

    return ensemble_model


def process_test_data(directory, filename, ex_generate_features):
    """
    Reads in the test data set and prepares it for prediction by the model.
    """
    test_data = load_csv_data(directory, filename)

    if ex_generate_features:
        test_data = generate_features(test_data)

    num_features = len(test_data.columns)
    X_test = test_data.iloc[:, 1:num_features].values

    return test_data, X_test


def create_submission(test_data, y_est, data_dir, submit_file):
    """
    Create a new submission file with test data and predictions generated by the model.
    """
    submit = pd.DataFrame(columns=['datetime', 'count'])
    submit['datetime'] = test_data['datetime']
    submit['count'] = y_est
    submit.to_csv(data_dir + submit_file, sep=',', index=False, index_label=False)


def main():
    ex_process_training_data = False
    ex_generate_features = False
    ex_create_transforms = False
    ex_load_model = False
    ex_save_model = False
    ex_visualize_feature_distributions = False
    ex_visualize_correlations = False
    ex_visualize_pairwise_relationships = False
    ex_visualize_sequential_relationships = False
    ex_visualize_regression_residuals = False
    ex_visualize_principal_components = False
    ex_train_model = False
    ex_visualize_feature_importance = False
    ex_cross_validate = False
    ex_plot_learning_curve = False
    ex_parameter_search = False
    ex_train_ensemble = False
    ex_create_submission = False

    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\BikeSharing\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\BikeSharing\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'

    model_type = 'classification'  # classification, regression
    algorithm = 'bayes'  # bayes, logistic, ridge, svm, sgd, forest, boost
    metric = None  # accuracy, f1, rcc_auc, mean_absolute_error, mean_squared_error, r2_score
    transform_list = ['scaler', 'pca', 'selector']
    transforms = {'scaler': None, 'pca': None, 'selector': None}
    column_offset = 3

    training_data = None
    X = None
    y1 = None
    y2 = None
    model = None

    os.chdir(code_dir)

    print('Starting process...')
    print('Algorithm = {0}'.format(algorithm))
    print('Scoring Metric = {0}'.format(metric))
    print('Generate Features = {0}'.format(ex_generate_features))
    print('Transforms = {0}'.format(transform_list))

    if ex_process_training_data:
        print('Reading in training data...')
        training_data, X, y1, y2 = process_training_data(data_dir, training_file, ex_generate_features)

    if ex_create_transforms:
        transforms = create_transforms(X, transform_list, transforms)

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

    if ex_visualize_regression_residuals:
        print('Visualizing feature distributions...')
        visualize_regression_residuals(training_data, X, y1, y2, None, None)

    if ex_visualize_principal_components:
        print('Visualizing feature distributions...')
        visualize_principal_components(training_data, X, y1, y2, None, None)

    if ex_load_model:
        print('Loading model from disk...')
        model = load_model(data_dir + model_file)

    if ex_train_model:
        print('Training model on full data set...')
        model = train_model(X, y1, model_type, algorithm, transforms)

        print('Calculating training score...')
        model_score = score(X, y1, model, transforms)
        print('Training score ='), model_score

        if ex_visualize_feature_importance and (algorithm == 'forest' or algorithm == 'boost'):
            print('Generating feature importance plot...')
            visualize_feature_importance(training_data, model, column_offset)

        if ex_cross_validate:
            print('Performing cross-validation...')
            cross_validation_score = cross_validate(X, y1, model_type, algorithm, metric, transforms)
            print('Cross-validation score ='), cross_validation_score

        if ex_plot_learning_curve:
            print('Generating learning curve...')
            plot_learning_curve(X, y1, model_type, algorithm, metric, transforms)

    if ex_parameter_search:
        print('Performing hyper-parameter grid search...')
        best_model, best_params, best_score = parameter_search(X, y1, model_type, algorithm, metric, transforms)
        print('Best model = ', best_model)
        print('Best params = ', best_params)
        print('Best score = ', best_score)

    if ex_train_ensemble:
        print('Creating an ensemble of models...')
        model = train_ensemble(X, y1, model_type, algorithm, transforms)

        print('Calculating ensemble training score...')
        ensemble_score = score(X, y1, model, transforms)
        print('Ensemble Training score ='), ensemble_score

    if ex_save_model:
        print('Saving model to disk...')
        save_model(model, data_dir + model_file)

    if ex_create_submission:
        print('Reading in test data...')
        test_data, X_test = process_test_data(data_dir, test_file, ex_generate_features)

        print('Predicting test data...')
        y_est = predict(X_test, model, transforms)

        print('Creating submission file...')
        create_submission(test_data, y_est, data_dir, submit_file)

    print('Process complete.')


if __name__ == "__main__":
    main()