import os
import math
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn import svm


def ams(s, b):
    """
    Approximate Median Significant function to evaluate solutions.
    """
    br = 10.0
    radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        print 'Radicand is negative.'
        exit()
    else:
        return math.sqrt(radicand)


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


def process_training_data(filename, features, impute, standardize, whiten):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv(filename, sep=',')
    
    # add a nominal label (0, 1)
    temp = training_data['Label'].replace(to_replace=['s', 'b'], value=[1, 0])
    training_data['Nominal'] = temp
    
    X = training_data.iloc[:, 1:features+1].values
    y = training_data.iloc[:, features+3].values
    w = training_data.iloc[:, features+1].values
    
    # optionally impute the -999 values
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X = imp.fit_transform(X)
    elif impute == 'zeros':
        X[X == -999] = 0
    
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
    
    return training_data, X, y, w, scaler, pca


def visualize(training_data, X, y, scaler, pca, features):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.
    
    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """
    
    # feature histograms
    fig1, ax1 = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(16):
        ax1[i % 4, i / 4].hist(X[:, i])
        ax1[i % 4, i / 4].set_title(training_data.columns[i + 1])
        ax1[i % 4, i / 4].set_xlim((min(X[:, i]), max(X[:, i])))
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(16, features):
        ax2[i % 4, (i - 16) / 4].hist(X[:, i])
        ax2[i % 4, (i - 16) / 4].set_title(training_data.columns[i + 1])
        ax2[i % 4, (i - 16) / 4].set_xlim((min(X[:, i]), max(X[:, i])))
    fig2.tight_layout()
    
    # covariance matrix
    if scaler is not None:
        X = scaler.transform(X)
        
    cov = np.cov(X, rowvar=0)
    
    fig3, ax3 = plt.subplots(figsize=(16, 10))
    p = ax3.pcolor(cov)
    fig3.colorbar(p, ax=ax3)
    ax3.set_title('Feature Covariance Matrix')
    
    # pca plots
    if pca is not None:
        X = pca.transform(X)
    
        fig4, ax4 = plt.subplots(figsize=(16, 10))
        ax4.scatter(X[:, 0], X[:, 1], c=y)
        ax4.set_title('First & Second Principal Components')
        
        fig5, ax5 = plt.subplots(figsize=(16, 10))
        ax5.scatter(X[:, 1], X[:, 2], c=y)
        ax5.set_title('Second & Third Principal Components')


def train(X, y, alg, scaler, pca):
    """
    Trains a new model using the training data.
    """     
    if scaler is not None:
        X = scaler.transform(X)
    
    if pca is not None:
        X = pca.transform(X)
    
    t0 = time.time()
    
    if alg == 'bayes':
        model = naive_bayes.GaussianNB()
    elif alg == 'logistic':
        model = linear_model.LogisticRegression()
    elif alg == 'svm':
        model = svm.SVC()
    elif alg == 'boost':
        model = ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=7, min_samples_split=200,
                                                    min_samples_leaf=200, max_features=30)
    else:
        print 'No model defined for ' + alg
        exit()
        
    model.fit(X, y)
    
    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)
    
    return model


def predict(X, model, threshold, scaler, pca):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    if pca is not None:
        X = pca.transform(X)
    
    y_prob = model.predict_proba(X)[:, 1]
    cutoff = np.percentile(y_prob, threshold)
    y_est = y_prob > cutoff

    return y_prob, y_est


def score(y, y_est, w):
    """
    Create weighted signal and background sets and calculate the AMS.
    """
    y_signal = w * (y == 1.0)
    y_background = w * (y == 0.0)
    s = np.sum(y_signal * (y_est == 1.0))
    b = np.sum(y_background * (y_est == 1.0))
    
    return ams(s, b)


def cross_validate(X, y, w, alg, scaler, pca, threshold):
    """
    Perform cross-validation on the training set and compute the AMS scores.
    """
    scores = [0, 0, 0]
    folds = cross_validation.StratifiedKFold(y, n_folds=3)
    i = 0
    
    for i_train, i_val in folds:
        # create the training and validation sets
        X_train, X_val = X[i_train], X[i_val]
        y_train, y_val = y[i_train], y[i_val]
        w_train, w_val = w[i_train], w[i_val]
        
        # normalize the weights   
        w_train[y_train == 1] *= (sum(w[y == 1]) / sum(w[y_train == 1]))
        w_train[y_train == 0] *= (sum(w[y == 0]) / sum(w_train[y_train == 0]))
        w_val[y_val == 1] *= (sum(w[y == 1]) / sum(w_val[y_val == 1]))
        w_val[y_val == 0] *= (sum(w[y == 0]) / sum(w_val[y_val == 0]))
        
        # train the model
        model = train(X_train, y_train, alg, scaler, pca)
        
        # predict and score performance on the validation set
        y_val_prob, y_val_est = predict(X_val, model, threshold, scaler, pca)
        scores[i] = score(y_val, y_val_est, w_val)
        i += 1
    
    return np.mean(scores)
    

def process_test_data(filename, features, impute):
    """
    Reads in test data and prepares numpy arrays.
    """
    test_data = pd.read_csv(filename, sep=',')
    X_test = test_data.iloc[:, 1:features+1].values
    
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X_test = imp.fit_transform(X_test)
    elif impute == 'zeros':
        X_test[X_test == -999] = 0
    
    return test_data, X_test


def create_submission(test_data, y_test_prob, y_test_est, submit_file):
    """
    Create a new data frame with the submission data.
    """
    temp = pd.DataFrame(y_test_prob, columns=['RankOrder'])
    temp2 = pd.DataFrame(y_test_est, columns=['Class'])
    submit = pd.DataFrame([test_data.EventId, temp.RankOrder, temp2.Class]).transpose()

    # sort it so they're in the ascending order by probability
    submit = submit.sort(['RankOrder'], ascending=True)
    
    # convert the probabilities to rank order (required by the submission guidelines)
    for i in range(0, y_test_est.shape[0], 1):
        submit.iloc[i, 1] = i + 1
    
    # re-sort by event ID
    submit = submit.sort(['EventId'], ascending=True)
    
    # convert the integer classification to (s, b)
    submit['Class'] = submit['Class'].map({1: 's', 0: 'b'})
    
    # force pandas to treat these columns at int (otherwise will write as floats)
    submit[['EventId', 'RankOrder']] = submit[['EventId', 'RankOrder']].astype(int)
    
    # finally create the submission file
    submit.to_csv(submit_file, sep=',', index=False, index_label=False)


def main():
    # perform some initialization
    features = 30
    threshold = 85
    alg = 'boost'  # bayes, logistic, svm, boost
    impute = 'none'  # zeros, mean, none
    standardize = False
    whiten = False
    load_training_data = True
    load_model = False
    train_model = False
    save_model = False
    create_visualizations = True
    create_submission_file = False
    code_dir = '/home/john/git/kaggle/HiggsBoson/'
    data_dir = '/home/john/data/higgs-boson/'
    training_file = 'training.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'
    
    os.chdir(code_dir)
    
    print 'Starting process...'
    print 'alg={0}, impute={1}, standardize={2}, whiten={3} threshold={4}'.format(
        alg, impute, standardize, whiten, threshold)
    
    if load_training_data:
        print 'Reading in training data...'
        training_data, X, y, w, scaler, pca = process_training_data(
            data_dir + training_file, features, impute, standardize, whiten)
    
    if create_visualizations:
        print 'Creating visualizations...'
        visualize(training_data, X, y, scaler, pca, features)
    
    if load_model:
        print 'Loading model from disk...'
        model = load(data_dir + model_file)
    
    if train_model:
        print 'Training model on full data set...'
        model = train(X, y, alg, scaler, pca)
        
        print 'Calculating predictions...'
        y_prob, y_est = predict(X, model, threshold, scaler, pca)
           
        print 'Calculating AMS...'
        ams_val = score(y, y_est, w)
        print 'AMS =', ams_val
        
        print 'Performing cross-validation...'
        val = cross_validate(X, y, w, alg, scaler, pca, threshold)
        print'Cross-validation AMS =', val
    
    if save_model:
        print 'Saving model to disk...'
        save(model, data_dir + model_file)
    
    if create_submission_file:
        print 'Reading in test data...'
        test_data, X_test = process_test_data(data_dir + test_file, features, impute)
        
        print 'Predicting test data...'
        y_test_prob, y_test_est = predict(X_test, model, threshold, scaler, pca)
        
        print 'Creating submission file...'
        create_submission(test_data, y_test_prob, y_test_est, data_dir + submit_file)
    
    print 'Process complete.'


if __name__ == "__main__":
    main()