from sklearn.cluster import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_extraction import *
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.manifold import *
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


def define_transforms():
    """
    Defines and returns a list of transforms.
    """
    transforms = [Imputer(missing_values='NaN', strategy='mean', axis=0),
                  LabelEncoder(),
                  OneHotEncoder(n_values='auto', categorical_features=[], sparse=False),
                  DictVectorizer(sparse=False),
                  FeatureHasher(n_features=1048576, input_type='dict'),
                  VarianceThreshold(threshold=0.0),
                  Binarizer(threshold=0.0),
                  StandardScaler(),
                  MinMaxScaler(),
                  PCA(n_components=None, whiten=False),
                  TruncatedSVD(n_components=None),
                  NMF(n_components=None),
                  FastICA(n_components=None, whiten=True),
                  Isomap(n_components=2),
                  LocallyLinearEmbedding(n_components=2, method='modified'),
                  MDS(n_components=2),
                  TSNE(n_components=2, learning_rate=1000, n_iter=1000),
                  KMeans(n_clusters=8)]

    return transforms


def define_keras_model(input_size, layer_size, output_size, n_hidden_layers, init_method, input_activation,
                       hidden_activation, output_activation, use_batch_normalization, input_dropout, hidden_dropout):
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

    return model


def get_keras_definition():
    """
    Defines and returns a Keras neural network model with the specified definition.
    """
    return define_keras_model(input_size=128,
                              layer_size=128,
                              output_size=1,
                              n_hidden_layers=2,
                              init_method='glorot_uniform',
                              input_activation='prelu',
                              hidden_activation='prelu',
                              output_activation='linear',
                              use_batch_normalization=True,
                              input_dropout=0.5,
                              hidden_dropout=0.5)


def define_model(task, algorithm):
    """
    Defines and returns a model object of the designated type.
    """
    if task == 'classification':
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
        elif algorithm == 'nn':
            model = KerasClassifier(model=get_keras_definition(), optimizer='adam', loss='categorical_crossentropy',
                                    train_batch_size=128, test_batch_size=128, nb_epoch=100, shuffle=True,
                                    show_accuracy=False, validation_split=0, validation_data=None, callbacks=None)
        else:
            raise Exception('No model defined for ' + algorithm)
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
            model = XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=1000, silent=True,
                                 objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
                                 subsample=1.0, colsample_bytree=1.0, base_score=0.5, seed=0, missing=None)
        elif algorithm == 'nn':
            model = KerasRegressor(model=get_keras_definition(), optimizer='adam', loss='mean_squared_error',
                                   train_batch_size=128, test_batch_size=128, nb_epoch=100, shuffle=True,
                                   show_accuracy=False, validation_split=0, validation_data=None, callbacks=None)
        else:
            raise Exception('No model defined for ' + algorithm)

    return model


def get_param_grid(algorithm):
    """
    Defines and returns a parameter grid for the designated model type.
    """
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
    elif algorithm == 'xgb':
        param_grid = [{'max_depth': [3, 5, 7, 9, None], 'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                       'n_estimators': [100, 300, 1000, 3000, 10000], 'min_child_weight': [1, 3, 5, 7, None],
                       'subsample': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 'colsample_bytree': [1.0, 0.9, 0.8, 0.7]}]
    elif algorithm == 'nn':
        param_grid = [{'layer_size': [64, 128, 256, 384, 512, 1024], 'n_hidden_layers': [1, 2, 3, 4, 5, 6],
                       'init_method': ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
                       'loss_function': ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'hinge',
                                         'binary_crossentropy', 'categorical_crossentropy'],
                       'input_activation': ['sigmoid', 'tanh', 'prelu', 'linear', 'softmax', 'softplus'],
                       'hidden_activation': ['sigmoid', 'tanh', 'prelu', 'linear', 'softmax', 'softplus'],
                       'output_activation': ['sigmoid', 'tanh', 'prelu', 'linear', 'softmax', 'softplus'],
                       'input_dropout': [0, 0.3, 0.5, 0.7], 'hidden_dropout': [0, 0.3, 0.5, 0.7],
                       'optimization_method': ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam'],
                       'batch_size': [16, 32, 64, 128, 256], 'nb_epoch': [10, 30, 100, 300, 1000]}]
    else:
        raise Exception('No params defined for ' + algorithm)

    return param_grid
