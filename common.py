"""
@author: John Wittenauer
"""

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


def define_transforms():
    """
    Defines and returns a list of transforms.
    """
    categories = []
    transforms = [Imputer(missing_values='NaN', strategy='mean', axis=0),
                  LabelEncoder(),
                  OneHotEncoder(n_values='auto', categorical_features=categories, sparse=False),
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

    transforms = [StandardScaler()]

    return transforms


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
            model = XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=1000, silent=True,
                                 objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
                                 subsample=1.0, colsample_bytree=1.0, base_score=0.5, seed=0, missing=None)
            model = BaggingRegressor(base_estimator=xg, n_estimators=10, max_samples=1.0, max_features=1.0,
                                     bootstrap=True, bootstrap_features=False)
        else:
            print('No model defined for ' + algorithm)
            exit()

    return model
