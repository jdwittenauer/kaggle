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


def generate_features(data):
    """
    Generates new derived features to add to the data set for model training.
    """
    data['Aspect_Shifted'] = data['Aspect'].map(lambda x: x - 180 if x + 180 < 360 else x + 180)
    data['High_Water'] = data['Vertical_Distance_To_Hydrology'] < 0
    data['EVDtH'] = data['Elevation'] - data['Vertical_Distance_To_Hydrology']
    data['EHDtH'] = data['Elevation'] - data['Horizontal_Distance_To_Hydrology'] * 0.2
    data['DTH'] = (data['Horizontal_Distance_To_Hydrology'] ** 2 + data['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    data['Hydro_Fire_1'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_Fire_2'] = abs(data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Fire_Points'])
    data['Hydro_Road_1'] = abs(data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways'])
    data['Hydro_Road_2'] = abs(data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Roadways'])
    data['Fire_Road_1'] = abs(data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Roadways'])
    data['Fire_Road_2'] = abs(data['Horizontal_Distance_To_Fire_Points'] - data['Horizontal_Distance_To_Roadways'])

    return data


def process_training_data(filename, create_features):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv(filename, sep=',')
    num_features = len(training_data.columns) - 1

    # move the label to the first position and drop the ID column
    cols = training_data.columns.tolist()
    cols = cols[-1:] + cols[1:num_features]
    training_data = training_data[cols]

    if create_features:
        training_data = generate_features(training_data)

    num_features = len(training_data.columns)
    X = training_data.iloc[:, 1:num_features].values
    y = training_data.iloc[:, 0].values

    return training_data, X, y


def create_transforms(X, standardize, whiten, select):
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

    # create a feature selection transform
    selector = None
    if select:
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(X)

    return scaler, pca, selector


def apply_transforms(X, scaler, pca, selector):
    """
    Applies pre-computed transformations to a data set.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if selector is not None:
        X = selector.transform(X)

    return X


def visualize(training_data, X, y, pca):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.

    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """

    print('Generating individual feature histograms...')
    num_features = len(training_data.columns)
    num_plots = num_features / 16 if num_features % 16 == 0 else num_features / 16 + 1
    for i in range(num_plots):
        fig, ax = plt.subplots(4, 4, figsize=(20, 10))
        for j in range(16):
            index = (i * 16) + j
            if index == 0:
                ax[j / 4, j % 4].hist(y, bins=30)
                ax[j / 4, j % 4].set_title(training_data.columns[index])
                ax[j / 4, j % 4].set_xlim((min(y), max(y)))
            elif index < num_features:
                ax[j / 4, j % 4].hist(X[:, index - 1], bins=30)
                ax[j / 4, j % 4].set_title(training_data.columns[index])
                ax[j / 4, j % 4].set_xlim((min(X[:, index - 1]), max(X[:, index - 1])))
        fig.tight_layout()

    print('Generating correlation matrix...')
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    colormap = sb.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF", "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
    sb.corrplot(training_data, annot=False, sig_stars=False, diag_names=False, cmap=colormap, ax=ax2)
    fig2.tight_layout()

    if pca is not None:
        print('Generating principal component plots...')
        X = pca.transform(X)
        class_count = np.count_nonzero(np.unique(y))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        fig3, ax3 = plt.subplots(figsize=(16, 10))
        for i in range(class_count):
            class_idx = i + 1  # add 1 if class labels start at 1 instead of 0
            ax3.scatter(X[y == class_idx, 0], X[y == class_idx, 1], c=colors[i], label=class_idx)
        ax3.set_title('First & Second Principal Components')
        ax3.legend()
        fig3.tight_layout()

        fig4, ax4 = plt.subplots(figsize=(16, 10))
        for i in range(class_count):
            class_idx = i + 1  # add 1 if class labels start at 1 instead of 0
            ax4.scatter(X[y == class_idx, 1], X[y == class_idx, 2], c=colors[i], label=class_idx)
        ax4.set_title('Second & Third Principal Components')
        ax4.legend()
        fig4.tight_layout()

        fig5, ax5 = plt.subplots(figsize=(16, 10))
        for i in range(class_count):
            class_idx = i + 1  # add 1 if class labels start at 1 instead of 0
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


def train(training_data, X, y, algorithm, scaler, pca, selector):
    """
    Trains a new model using the training data.
    """
    t0 = time.time()
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca, selector)
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


def predict(X, model, scaler, pca, selector):
    """
    Predicts the class label.
    """
    X = apply_transforms(X, scaler, pca, selector)
    y_est = model.predict(X)

    return y_est


def predict_probability(X, model, scaler, pca, selector):
    """
    Predicts the class probabilities.
    """
    X = apply_transforms(X, scaler, pca, selector)
    y_prob = model.predict_proba(X)[:, 1]

    return y_prob


def score(X, y, model, scaler, pca, selector):
    """
    Scores the model's performance and returns the result.
    """
    X = apply_transforms(X, scaler, pca, selector)

    return model.score(X, y)


def cross_validate(X, y, algorithm, scaler, pca, selector, metric):
    """
    Performs cross-validation to estimate the true performance of the model.
    """
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca, selector)

    t0 = time.time()
    scores = cross_validation.cross_val_score(model, X, y, scoring=metric, cv=3, n_jobs=-1)
    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def plot_learning_curve(X, y, algorithm, scaler, pca, selector, metric):
    """
    Plots a learning curve showing model performance against both training and
    validation data sets as a function of the number of training samples.
    """

    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca, selector)

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


def parameter_search(X, y, algorithm, scaler, pca, selector, metric):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca, selector)

    param_grid = None
    if algorithm == 'logistic':
        param_grid = [{'penalty': ['l1', 'l2'], 'C': [0.1, 0.3, 1.0, 3.0]}]
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


def train_ensemble(X, y, algorithm, scaler, pca, selector):
    """
    Creates an ensemble of many models together.
    """
    model = define_model(algorithm)
    X = apply_transforms(X, scaler, pca, selector)

    t0 = time.time()
    ensemble_model = BaggingClassifier(base_estimator=model, n_estimators=10, max_samples=1.0, max_features=1.0,
                                       bootstrap=True, bootstrap_features=False)
    ensemble_model.fit(X, y)
    t1 = time.time()
    print('Ensemble training completed in {0:3f} s.'.format(t1 - t0))

    return ensemble_model


def process_test_data(filename, create_features):
    """
    Reads in the test data set and prepares it for prediction by the model.
    """
    test_data = pd.read_csv(filename, sep=',')

    if create_features:
        test_data = generate_features(test_data)

    num_features = len(test_data.columns)
    X_test = test_data.iloc[:, 1:num_features].values

    return test_data, X_test


def create_submission(test_data, y_est, submit_file):
    """
    Create a new submission file with test data and predictions generated by the model.
    """
    submit = pd.DataFrame(columns=['Id', 'Cover_Type'])
    submit['Id'] = test_data['Id']
    submit['Cover_Type'] = y_est
    submit.to_csv(submit_file, sep=',', index=False, index_label=False)


def main():
    load_training_data = True
    create_features = False
    create_visualizations = False
    load_model = False
    train_model = True
    create_learning_curve = False
    perform_grid_search = False
    perform_ensemble = False
    save_model = False
    create_submission_file = False

    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\ForestCover\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\ForestCover\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'

    algorithm = 'forest'  # bayes, logistic, svm, sgd, forest, boost
    metric = None  # accuracy, f1, rcc_auc, mean_absolute_error, mean_squared_error, r2_score
    standardize = False
    whiten = False
    select = True

    training_data = None
    X = None
    y = None
    scaler = None
    pca = None
    selector = None
    model = None
    ensemble_model = None

    os.chdir(code_dir)

    print('Starting process...')
    print('Algorithm={0}, Create={1}, Select={2}, Standardize={3}, Whiten={4}'.format(
        algorithm, create_features, select, standardize, whiten))

    if load_training_data:
        print('Reading in training data...')
        training_data, X, y = process_training_data(data_dir + training_file, create_features)

    if standardize or whiten or select:
        print('Creating data transforms...')
        scaler, pca, selector = create_transforms(X, standardize, whiten, select)

    if create_visualizations:
        print('Creating visualizations...')
        visualize(training_data, X, y, pca)

    if load_model:
        print('Loading model from disk...')
        model = load(data_dir + model_file)

    if train_model:
        print('Training model on full data set...')
        model = train(training_data, X, y, algorithm, scaler, pca, selector)

        print('Calculating training score...')
        model_score = score(X, y, model, scaler, pca, selector)
        print('Training score ='), model_score

        if create_learning_curve:
            print('Generating learning curve...')
            plot_learning_curve(X, y, algorithm, scaler, pca, selector, metric)
        else:
            print('Performing cross-validation...')
            cross_val_score = cross_validate(X, y, algorithm, scaler, pca, selector, metric)
            print('Cross-validation score ='), cross_val_score

    if perform_grid_search:
        print('Performing hyper-parameter grid search...')
        best_model, best_params, best_score = parameter_search(X, y, algorithm, scaler, pca, selector, metric)
        print('Best model = ', best_model)
        print('Best params = ', best_params)
        print('Best score = ', best_score)

    if perform_ensemble:
        print('Creating an ensemble of models...')
        ensemble_model = train_ensemble(X, y, algorithm, scaler, pca, selector)

        print('Calculating ensemble training score...')
        ensemble_model_score = score(X, y, ensemble_model, scaler, pca, selector)
        print('Ensemble Training score ='), ensemble_model_score

    if save_model:
        print('Saving model to disk...')
        save(model, data_dir + model_file)

    if create_submission_file:
        print('Reading in test data...')
        test_data, X_test = process_test_data(data_dir + test_file, create_features)

        print('Predicting test data...')
        if perform_ensemble:
            y_est = predict(X_test, ensemble_model, scaler, pca, selector)
        else:
            y_est = predict(X_test, model, scaler, pca, selector)

        print('Creating submission file...')
        create_submission(test_data, y_est, data_dir + submit_file)

    print('Process complete.')


if __name__ == "__main__":
    main()