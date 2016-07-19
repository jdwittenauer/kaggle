import os
import time
import pandas as pd
import numpy as np

from sklearn.cross_validation import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.ensemble import *

import xgboost as xgb

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils


def predict_probability(X, model, scaler):
    X = apply_scaler(X, scaler)
    y_prob = model.predict_proba(X)

    return y_prob


def score(X, y, model, scaler):
    X = apply_scaler(X, scaler)
    y_est = model.predict_proba(X)

    return log_loss(y, y_est)


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
    return scaler.transform(X)


def preprocess_labels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    y_onehot = np_utils.to_categorical(y)

    return y, y_onehot, encoder


def define_xgb_model():
    model = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=250, silent=True,
                              objective="multi:softprob", nthread=-1, gamma=0, min_child_weight=4,
                              max_delta_step=0, subsample=0.9, colsample_bytree=0.8, base_score=0.5, seed=0)

    return model


def define_nn_model(num_features, num_classes):
    layer_size = 512
    init_method = 'glorot_uniform'
    loss_function = 'categorical_crossentropy'
    optimization_method = 'adam'

    model = Sequential()
    model.add(Dense(num_features, layer_size, init=init_method))
    model.add(PReLU((layer_size,)))
    model.add(BatchNormalization((layer_size,)))
    model.add(Dropout(0.5))

    model.add(Dense(layer_size, layer_size, init=init_method))
    model.add(PReLU((layer_size,)))
    model.add(BatchNormalization((layer_size,)))
    model.add(Dropout(0.5))

    model.add(Dense(layer_size, layer_size, init=init_method))
    model.add(PReLU((layer_size,)))
    model.add(BatchNormalization((layer_size,)))
    model.add(Dropout(0.5))

    model.add(Dense(layer_size, num_classes, init=init_method))
    model.add(Activation('softmax'))

    model.compile(loss=loss_function, optimizer=optimization_method)

    return model


def train_xgb_model(X, y, model, scaler):
    t0 = time.time()
    X = apply_scaler(X, scaler)
    model.fit(X, y)
    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    return model


def train_nn_model(X, y_onehot, model, scaler):
    t0 = time.time()
    X = apply_scaler(X, scaler)
    model.fit(X, y_onehot, nb_epoch=20, batch_size=16, verbose=0)
    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    return model


def cross_validate_xgb(X, y, scaler, folds=3):
    model = define_xgb_model()
    X = apply_scaler(X, scaler)
    t0 = time.time()

    scores = []
    kf = KFold(y.shape[0], n_folds=folds, shuffle=True)
    for train_index, test_index in kf:
        model.fit(X[train_index], y[train_index])
        predictions = model.predict_proba(X[test_index])
        actuals = y[test_index]
        scores.append(log_loss(actuals, predictions))

    t1 = time.time()
    print('Cross-validation completed in {0:3f} s.'.format(t1 - t0))

    return np.mean(scores)


def cross_validate_nn(X, y, y_onehot, scaler, num_features, num_classes, folds=3):
    model = define_nn_model(num_features, num_classes)
    X = apply_scaler(X, scaler)
    t0 = time.time()

    scores = []
    kf = KFold(y.shape[0], n_folds=folds, shuffle=True)
    for train_index, test_index in kf:
        model.fit(X[train_index], y_onehot[train_index], nb_epoch=20, batch_size=16, verbose=0)
        predictions = model.predict_proba(X[test_index])
        actuals = y[test_index]
        scores.append(log_loss(actuals, predictions))

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
    code_dir = '/home/git/kaggle/OttoGroup/'
    data_dir = '/home/data/otto-group/'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'

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
    model = define_xgb_model()

    print('Training model...')
    model = train_xgb_model(X, y, model, scaler)

    print('Training score = ' + str(score(X, y, model, scaler)))

    print('Running cross-validation...')
    val_score = cross_validate_xgb(X, y,  scaler)
    print('Cross-validation score = ' + str(val_score))

    print('Building ensemble...')
    ensemble = BaggingClassifier(model, n_estimators=5, max_samples=1.0, max_features=1.0)

    print('Training ensemble...')
    X = apply_scaler(X, scaler)
    ensemble.fit(X, y)

    print('Generating submission file...')
    y_prob = predict_probability(X_test, ensemble, scaler)
    make_submission(y_prob, ids, encoder, data_dir, submit_file)

    print('Script complete.')


if __name__ == "__main__":
    main()
