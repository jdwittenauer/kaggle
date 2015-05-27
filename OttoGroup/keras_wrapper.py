import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import *
from sklearn.ensemble import *

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier


def load_training_data(path, filename):
    df = pd.read_csv(path + filename)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]

    return X, labels


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


def define_model(num_features, num_classes):
    layer_size = 512
    init_method = 'glorot_uniform'

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

    return model


def main():
    code_dir = '/home/john/git/kaggle/OttoGroup/'
    data_dir = '/home/john/data/otto/'
    training_file = 'train.csv'

    os.chdir(code_dir)
    np.random.seed(1337)

    print('Starting script...')

    print('Loading data...')
    X, labels = load_training_data(data_dir, training_file)

    print('Pre-processing...')
    scaler = create_scaler(X)
    X = apply_scaler(X, scaler)
    y, y_onehot, encoder = preprocess_labels(labels)
    num_features = X.shape[1]
    num_classes = y_onehot.shape[1]
    print('Features = ' + str(num_features))
    print('Classes = ' + str(num_classes))

    print('Building model...')
    model = define_model(num_features, num_classes)
    print('Complete.')

    print('Training model...')
    wrapper = KerasClassifier(model)
    wrapper.fit(X, y_onehot, nb_epoch=20)
    print('Complete.')

    print('Training score = ' + str(wrapper.score(X, y_onehot)))

    preds = wrapper.predict(X)
    print('Predictions shape = ' + str(preds.shape))

    proba = wrapper.predict_proba(X)
    print('Probabilities shape = ' + str(proba.shape))

    print('Building ensemble...')
    ensemble = BaggingClassifier(wrapper, n_estimators=3, max_samples=1.0, max_features=1.0)
    print('Complete.')

    print('Training ensemble...')
    ensemble.fit(X, y)
    print('Complete.')

    print('Ensemble score = ' + str(ensemble.score(X, y)))

    print('Script complete.')


if __name__ == "__main__":
    main()
