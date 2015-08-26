"""
@author: John Wittenauer
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime

from sklearn.cluster import *
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_extraction import *
from sklearn.feature_selection import *
from sklearn.grid_search import *
from sklearn.learning_curve import *
from sklearn.linear_model import *
from sklearn.manifold import *
from sklearn.metrics import *
from sklearn.naive_bayes import *
from sklearn.preprocessing import *
from sklearn.svm import *

from xgboost import *
from keras.callbacks import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.layers.advanced_activations import *
from keras.models import *
from keras.optimizers import *

from ionyx.utils import *


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()


def load_csv_data(directory, filename, dtype=None, index=None, convert_to_date=False):
    """
    Load a csv data file into a data frame, setting the index as appropriate.
    """
    data = pd.read_csv(directory + filename, sep=',', dtype=dtype)

    if index is not None:
        if convert_to_date:
            if type(index) is str:
                data[index] = data[index].convert_objects(convert_dates='coerce')
            else:
                for key in index:
                    data[key] = data[key].convert_objects(convert_dates='coerce')

        data = data.set_index(index)

    print('Data file ' + filename + ' loaded successfully.')

    return data


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
        elif algorithm == 'xt':
            model = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', max_depth=None,
                                         min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'boost':
            model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features=None,
                                               max_leaf_nodes=None)
        elif algorithm == 'xgb':
            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                                  objective='multi:softmax', gamma=0, min_child_weight=1, max_delta_step=0,
                                  subsample=1.0, colsample_bytree=1.0, base_score=0.5, seed=0, missing=None)
        else:
            print('No model defined for ' + algorithm)
            exit()
    else:
        if algorithm == 'ridge':
            model = Ridge(alpha=1.0)
        elif algorithm == 'svm':
            model = SVR(C=1.0, kernel='rbf', shrinking=True, cache_size=200)
        elif algorithm == 'sgd':
            model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, n_iter=1000, shuffle=False)
        elif algorithm == 'forest':
            model = RandomForestRegressor(n_estimators=10, criterion='mse', max_features='auto', max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'xt':
            model = ExtraTreesRegressor(n_estimators=10, criterion='mse', max_features='auto', max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'boost':
            model = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                              min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features=None,
                                              max_leaf_nodes=None)
        elif algorithm == 'xgb':
            # model = XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=1000, silent=True,
            #                      objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
            #                      subsample=1.0, colsample_bytree=1.0, base_score=0.5, seed=0, missing=None)
            xg = XGBRegressor(max_depth=7, learning_rate=0.005, n_estimators=1800, silent=True,
                              objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
                              subsample=0.9, colsample_bytree=0.8, base_score=0.5, seed=0, missing=None)
            model = BaggingRegressor(base_estimator=xg, n_estimators=10, max_samples=1.0, max_features=1.0,
                                     bootstrap=True, bootstrap_features=False)
        else:
            print('No model defined for ' + algorithm)
            exit()

    return model


def define_nn_model(input_size, layer_size, output_size, n_hidden_layers, init_method, loss_function,
                             input_activation, hidden_activation, output_activation, use_batch_normalization,
                             input_dropout, hidden_dropout, optimization_method):
    """
    Defines and returns a Keras neural network model.
    """
    model = Sequential()

    # add input layer
    model.add(Dense(input_size, layer_size, init=init_method))

    if input_activation == 'prelu':
        model.add(PReLU((layer_size,)))
    else:
        model.add(Activation(input_activation))

    if use_batch_normalization:
        model.add(BatchNormalization((layer_size,)))

    model.add(Dropout(input_dropout))

    # add hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(layer_size, layer_size, init=init_method))

        if hidden_activation == 'prelu':
            model.add(PReLU((layer_size,)))
        else:
            model.add(Activation(hidden_activation))

        if use_batch_normalization:
            model.add(BatchNormalization((layer_size,)))

        model.add(Dropout(hidden_dropout))

    # add output layer
    model.add(Dense(layer_size, output_size, init=init_method))
    model.add(Activation(output_activation))

    # configure optimization method
    if optimization_method == 'sgd':
        optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True)
    elif optimization_method == 'adagrad':
        optimizer = Adagrad()
    elif optimization_method == 'adadelta':
        optimizer = Adadelta()
    elif optimization_method == 'rmsprop':
        optimizer = RMSprop()
    elif optimization_method == 'adam':
        optimizer = Adam()
    else:
        raise Exception('Optimization method not recognized.')

    model.compile(loss=loss_function, optimizer=optimizer)

    return model


def train_model(X, y, algorithm, model, metric, transforms, early_stopping):
    """
    Trains a new model using the training data.
    """
    if algorithm == 'xgb':
        return train_xgb_model(X, y, model, metric, transforms, early_stopping)
    elif algorithm == 'nn':
        return train_nn_model(X, y, model, metric, transforms, early_stopping)
    else:
        t0 = time.time()
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        model.fit(X, y)
        t1 = time.time()
        print('Model trained in {0:3f} s.'.format(t1 - t0))

        print('Model hyper-parameters:')
        print(model.get_params())

        print('Calculating training score...')
        model_score = predict_score(X, y, model, metric)
        print('Training score ='), model_score

        return model


def train_xgb_model(X, y, model, metric, transforms, early_stopping):
    """
    Trains a new model XGB using the training data.
    """
    t0 = time.time()

    if early_stopping:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                  early_stopping_rounds=100)
        print('Best iteration found: ' + str(model.best_iteration))

        print('Re-fitting at the new stopping point...')
        model.n_estimators = model.best_iteration
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        model.fit(X, y)
    else:
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        model.fit(X, y)

    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    print('Model hyper-parameters:')
    print(model.get_params())

    print('Calculating training score...')
    model_score = predict_score(X, y, model, metric)
    print('Training score ='), model_score

    return model


def train_nn_model(X, y, model, metric, transforms, early_stopping):
    """
    Trains a new Keras model using the training data.
    """
    t0 = time.time()
    X_train = None
    X_eval = None
    y_train = None
    y_eval = None

    print('Beginning training...')
    if early_stopping:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)
        # eval_monitor = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        # history = model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=0,
        #                     validation_data=(X_eval, y_eval), shuffle=True, callbacks=[eval_monitor])
        history = model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=0,
                            validation_data=(X_eval, y_eval), shuffle=True)
    else:
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        history = model.fit(X, y, batch_size=128, nb_epoch=100, verbose=0, shuffle=True, callbacks=[])

    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    print('Model hyper-parameters:')
    print(model.get_config())

    print('Min eval loss ='), min(history.history['val_loss'])
    print('Min eval epoch ='), min(enumerate(history.history['loss']), key=lambda x: x[1])[0] + 1

    if early_stopping:
        print('Calculating training score...')
        train_score = predict_score(X_train, y_train, model, metric)
        print('Training score ='), train_score

        print('Calculating evaluation score...')
        eval_score = predict_score(X_eval, y_eval, model, metric)
        print('Evaluation score ='), eval_score
    else:
        print('Calculating training score...')
        train_score = predict_score(X, y, model, metric)
        print('Training score ='), train_score

    return model, history


def visualize_feature_importance(train_data, model, column_offset):
    """
    Generates a feature importance plot.  Requires a trained random forest or gradient boosting model.
    Does not work properly if transformations are applied to training data that expands the number
    of features.
    """
    importance = model.feature_importances_
    importance = 100.0 * (importance / importance.max())
    importance = importance[0:30] if len(train_data.columns) > 30 else importance
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0])

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Variable Importance')
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(train_data.columns[sorted_idx + column_offset])
    ax.set_xlabel('Relative Importance')

    fig.tight_layout()


def cross_validate(X, y, algorithm, model, metric, transforms, n_folds):
    """
    Performs manual cross-validation to estimate the true performance of the model.
    """
    t0 = time.time()
    y_pred = np.array([])
    y_true = np.array([])

    folds = list(KFold(y.shape[0], n_folds=n_folds, shuffle=True, random_state=1337))
    for i, (train_index, eval_index) in enumerate(folds):
        print('Starting fold {0}...'.format(i + 1))
        X_train = X[train_index]
        y_train = y[train_index]
        X_eval = X[eval_index]
        y_eval = y[eval_index]

        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        if algorithm == 'nn':
            model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=0,
                      validation_data=(X_eval, y_eval), shuffle=True)
        elif algorithm == 'xgb':
            model.fit(X_train, y_train)

        y_pred = np.append(y_pred, model.predict(X_eval))
        y_true = np.append(y_true, y_eval)

    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    cross_validation_score = score(y_true, y_pred, metric)
    print('Cross-validation score ='), cross_validation_score

    return cross_validation_score


def sequence_cross_validate(X, y, model, metric, transforms, n_folds, strategy='traditional',
                            window_type='cumulative', min_window=0, forecast_range=1, plot=False):
    """
    Performs time series cross-validation to estimate the true performance of the model.
    """
    scores = []
    train_count = len(X)

    if strategy == 'walk-forward':
        n_folds = train_count - min_window - forecast_range
        fold_size = 1
    else:
        fold_size = train_count / n_folds

    t0 = time.time()
    for i in range(n_folds):
        if window_type == 'fixed':
            fold_start = i * fold_size
        else:
            fold_start = 0

        fold_end = (i + 1) * fold_size + min_window
        fold_train_end = fold_end - forecast_range

        X_train, X_eval = X[fold_start:fold_train_end, :], X[fold_train_end:fold_end, :]
        y_train, y_eval = y[fold_start:fold_train_end], y[fold_train_end:fold_end]

        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)
        scores.append(score(y, y_pred, metric))

        if plot is True:
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.set_title('Estimation Error')
            ax.plot(y_pred - y_eval)
            fig.tight_layout()

    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def plot_learning_curve(X, y, model, metric, transforms, n_folds):
    """
    Plots a learning curve showing model performance against both training and
    validation data sets as a function of the number of training samples.
    """
    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)

    t0 = time.time()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring=metric, cv=n_folds, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    alpha=0.1, color='b')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                    alpha=0.1, color='r')
    ax.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation score')
    ax.legend(loc='best')
    fig.tight_layout()
    t1 = time.time()
    print('Learning curve generated in {0:3f} s.'.format(t1 - t0))


def parameter_search(X, y, algorithm, model, metric, transforms, n_folds):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    if algorithm == 'xgb':
        xbg_parameter_search(X, y, metric)
    elif algorithm == 'nn':
        nn_parameter_search(X, y, metric)
    else:
        transforms = fit_transforms(X, y, transforms)
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
        elif algorithm == 'forest' or algorithm == 'xt':
            param_grid = [{'n_estimators': [10, 30, 100, 300], 'criterion': ['gini', 'entropy', 'mse'],
                           'max_features': ['auto', 'log2', None], 'max_depth': [3, 5, 7, 9, None],
                           'min_samples_split': [2, 10, 30, 100], 'min_samples_leaf': [1, 3, 10, 30, 100]}]
        elif algorithm == 'boost':
            param_grid = [{'learning_rate': [0.1, 0.3, 1.0], 'subsample': [1.0, 0.9, 0.7, 0.5],
                           'n_estimators': [100, 300, 1000], 'max_features': ['auto', 'log2', None],
                           'max_depth': [3, 5, 7, 9, None], 'min_samples_split': [2, 10, 30, 100],
                           'min_samples_leaf': [1, 3, 10, 30, 100]}]

        t0 = time.time()
        grid_estimator = GridSearchCV(model, param_grid, scoring=metric, cv=n_folds, n_jobs=1)
        grid_estimator.fit(X, y)
        t1 = time.time()
        print('Grid search completed in {0:3f} s.'.format(t1 - t0))

        print('Best params ='), grid_estimator.best_params_
        print('Best score ='), grid_estimator.best_score_


def xbg_parameter_search(X, y, metric):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    categories = []
    # categories = [3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 19, 21, 27, 28, 29]
    # categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
    #               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    transforms = [OneHotEncoder(n_values='auto', categorical_features=categories, sparse=False),
                  StandardScaler()]

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
    transforms = fit_transforms(X_train, y_train, transforms)
    X_train = apply_transforms(X_train, transforms)
    X_eval = apply_transforms(X_eval, transforms)

    for subsample in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        for colsample_bytree in [1.0, 0.9, 0.8, 0.7]:
            for max_depth in [3, 5, 7, 9]:
                for min_child_weight in [1, 3, 5, 7]:
                    t0 = time.time()
                    model = XGBRegressor(max_depth=max_depth, learning_rate=0.005, n_estimators=5000, silent=True,
                                         objective='reg:linear', gamma=0, min_child_weight=min_child_weight,
                                         max_delta_step=0, subsample=subsample, colsample_bytree=colsample_bytree,
                                         base_score=0.5, seed=0, missing=None)

                    print('subsample ='), subsample
                    print('colsample_bytree ='), colsample_bytree
                    print('max_depth ='), max_depth
                    print('min_child_weight ='), min_child_weight

                    print('Model hyper-parameters:')
                    print(model.get_params())

                    print('Fitting model...')
                    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                              early_stopping_rounds=100, verbose=False)
                    print('Best iteration ='), model.best_iteration

                    train_score = predict_score(X_train, y_train, model, metric)
                    print('Training score ='), train_score

                    eval_score = predict_score(X_eval, y_eval, model, metric)
                    print('Evaluation score ='), eval_score

                    t1 = time.time()
                    print('Model trained in {0:3f} s.'.format(t1 - t0))
                    print('')
                    print('')


def nn_parameter_search(X, y, metric):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    categories = [3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 19, 21, 27, 28, 29]
    transforms = [FactorToNumeric(categorical_features=categories, metric='mean'),
                  StandardScaler()]

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
    transforms = fit_transforms(X_train, y_train, transforms)
    X_train = apply_transforms(X_train, transforms)
    X_eval = apply_transforms(X_eval, transforms)

    # init_methods = ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    # optimization_methods = ['adagrad', 'adadelta', 'rmsprop', 'adam']
    # layer_sizes = [64, 128, 256, 384, 512]
    # hidden_layers = [1, 2, 3, 4]
    # batch_sizes = [16, 32, 64, 128]

    # init_methods = ['glorot_uniform']
    # optimization_methods = ['adadelta', 'rmsprop']
    # layer_sizes = [64, 128, 192]
    # hidden_layers = [1, 2, 3, 4]
    # batch_sizes = [128]

    init_methods = ['glorot_uniform']
    optimization_methods = ['adadelta']
    layer_sizes = [64, 128, 256]
    hidden_layers = [1, 2, 3, 4]
    batch_sizes = [128]

    for init_method in init_methods:
        for optimization_method in optimization_methods:
            for layer_size in layer_sizes:
                for hidden_layer in hidden_layers:
                    for batch_size in batch_sizes:
                        t0 = time.time()
                        print('Compiling model...')
                        model = define_nn_model_detailed(input_size=X_train.shape[1],
                                                         layer_size=layer_size,
                                                         output_size=1,
                                                         n_hidden_layers=hidden_layer,
                                                         init_method=init_method,
                                                         loss_function='mse',
                                                         input_activation='prelu',
                                                         hidden_activation='prelu',
                                                         output_activation='linear',
                                                         use_batch_normalization=True,
                                                         input_dropout=0.5,
                                                         hidden_dropout=0.5,
                                                         optimization_method=optimization_method)

                        print('init_method ='), init_method
                        print('optimization_method ='), optimization_method
                        print('layer_size ='), layer_size
                        print('hidden_layer ='), hidden_layer
                        print('batch_size ='), batch_size

                        print('Model hyper-parameters:')
                        print(model.get_config())

                        print('Fitting model...')
                        # eval_monitor = EarlyStopping(monitor='val_loss', patience=100, verbose=0)
                        # history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, verbose=0,
                        #                     validation_data=(X_eval, y_eval), shuffle=True, callbacks=[eval_monitor])
                        history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, verbose=0,
                                            validation_data=(X_eval, y_eval), shuffle=True)
                        print('Min eval loss ='), min(history.history['val_loss'])
                        print('Min eval epoch ='), min(enumerate(history.history['loss']), key=lambda x: x[1])[0] + 1

                        train_score = predict_score(X_train, y_train, model, metric)
                        print('Training score ='), train_score

                        eval_score = predict_score(X_eval, y_eval, model, metric)
                        print('Evaluation score ='), eval_score

                        t1 = time.time()
                        print('Model trained in {0:3f} s.'.format(t1 - t0))
                        print('')
                        print('')
