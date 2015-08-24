import pandas as pd
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os


def log_loss_func(weights):
    """
    scipy minimize will pass the weights as a numpy array
    """
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(test_y, final_prediction)


os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

print(train.head())

# we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(labels, test_size=0.05, random_state=1234)
for train_index, test_index in sss:
    break

train_x, train_y = train.values[train_index], labels.values[train_index]
test_x, test_y = train.values[test_index], labels.values[test_index]

# building the classifiers
clfs = []

rfc = RandomForestClassifier(n_estimators=50, random_state=4141, n_jobs=-1)
rfc.fit(train_x, train_y)
print('RFC LogLoss {score}'.format(score=log_loss(test_y, rfc.predict_proba(test_x))))
clfs.append(rfc)

# usually you'd use xgboost and neural nets here
logreg = LogisticRegression()
logreg.fit(train_x, train_y)
print('LogisticRegression LogLoss {score}'.format(score=log_loss(test_y, logreg.predict_proba(test_x))))
clfs.append(logreg)

rfc2 = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
rfc2.fit(train_x, train_y)
print('RFC2 LogLoss {score}'.format(score=log_loss(test_y, rfc2.predict_proba(test_x))))
clfs.append(rfc2)


# finding the optimum weights
predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(test_x))

# the algorithms need a starting value, right not we chose 0.5 for all weights
# its better to choose many random starting points and run minimize a few times
starting_values = [0.5] * len(predictions)

# adding constraints and a different solver as suggested by user 16universe
cons = ({'type': 'eq', 'fun': lambda w: 1-sum(w)})

# our weights are bound between 0 and 1
bounds = [(0, 1)] * len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
