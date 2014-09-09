# -*- coding: utf-8 -*-
"""
@author: John Wittenauer

@notes: This script was tested on 64-bit Ububtu 14 using the Anaconda 2.0
distribution of 64-bit Python 2.7.
"""


def AMS(s, b):
    """
    Approximate Median Significant function to evaluate solutions.
    """
    br = 10.0
    radicand = 2 *( (s + b + br) * math.log(1.0 + s / (b + br)) - s)
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


def save(filename):
    """
    Persist a trained model to disk.
    """
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def processTrainingData(filename, features, impute, standardize, whiten):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv(filename, sep=',')
    
    # add a nominal label (0, 1)
    temp = training_data['Label'].replace(to_replace=['s','b'], value=[1,0])
    training_data['Nominal'] = temp
    
    X = training_data.iloc[:,1:features].values
    y = training_data.iloc[:,features+2].values
    w = training_data.iloc[:,features].values
    
    # optionally impute the -999 values
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X = imp.fit_transform(X)
    elif impute == 'zeros':
        X[X == -999] = 0
    
    # create a standardization transform
    scaler = None
    if standardize == True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
    
    # create a PCA transform
    pca = None
    if whiten == True:
        pca = decomposition.PCA(whiten=True)
        pca.fit(X)
    
    return training_data, X, y, w, scaler, pca


def visualize(training_data, X, y, scaler, pca):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.
    
    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """
    # TODO - add visualizations
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Title')


def train(X, y, w, alg, scaler, pca):
    """
    Trains a new model using the training data.
    """
    if scaler != None:
        X = scaler.transform(X)
    
    if pca != None:
        X = pca.transform(X)    
    
    if alg == 'xgboost':
        # use a separate process for the xgboost library
        return trainXGB(X, y, w, scaler, pca)
    
    t0 = time.time()
    
    if alg == 'bayes':
        model = naive_bayes.GaussianNB()
    elif alg == 'logistic':
        model = linear_model.LogisticRegression()
    elif alg == 'svm':
        model = svm.SVC()
    elif alg == 'boost':
        model = ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=7,
            min_samples_split=200, min_samples_leaf=200, max_features=30)
    else:
        print 'No model defined for ' + alg
        exit()
        
    model.fit(X, y)
    
    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)
    
    #TODO - add feature importance visualization
    
    return model


def trainXGB(X, y, w, scaler, pca):
    """
    Trains a boosted trees model using the XGBoost library.
    """
    t0 = time.time()
    
    xgmat = xgb.DMatrix(X, label=y, missing=-999.0, weight=w)
    
    w_pos = sum(w[i] for i in range(len(y)) if y[i] == 1)
    w_neg = sum(w[i] for i in range(len(y)) if y[i] == 0)
    
    param = {}
    param['objective'] = 'binary:logitraw'
    param['scale_pos_weight'] = w_neg/w_pos
    param['eta'] = 0.1 
    param['max_depth'] = 7
    param['subsample'] = 0.8
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    
    plst = list(param.items())
    watchlist = [ ]
    
    model = xgb.train(plst, xgmat, 130, watchlist)
    
    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)

    return model


def predict(X, model, alg, threshold, scaler, pca):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    if scaler != None:
        X = scaler.transform(X)
    
    if pca != None:
        X = pca.transform(X)
    
    if alg == 'xgboost':
        xgmat = xgb.DMatrix(X, missing=-999.0)
        y_prob = model.predict(xgmat)
    else:
        y_prob = model.predict_proba(X)[:,1]  
    
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
    
    return AMS(s, b)


def crossValidate(X, y, w, alg, scaler, pca, threshold):
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
        model = train(X_train, y_train, w_train, alg, scaler, pca)
        
        # predict and score preformance on the validation set
        y_val_prob, y_val_est = predict(X_val, model, alg, threshold, scaler, pca)
        scores[i] = score(y_val, y_val_est, w_val)
        i = i + 1
    
    return np.mean(scores)
    

def processTestData(filename, features, impute):
    """
    Reads in test data and prepares numpy arrays.
    """
    test_data = pd.read_csv(filename, sep=',')
    X_test = test_data.iloc[:,1:features].values
    
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X_test = imp.fit_transform(X_test)
    elif impute == 'zeros':
        X[X == -999] = 0
    
    return test_data, X_test


def createSubmission(test_data, y_test_prob, y_test_est):
    """
    Create a new datafrane with the submission data.
    """
    temp = pd.DataFrame(y_test_prob, columns=['RankOrder'])
    temp2 = pd.DataFrame(y_test_est, columns=['Class'])
    submit = pd.DataFrame([test_data.EventId, temp.RankOrder, temp2.Class]).transpose()

    # sort it so they're in the ascending order by probability
    submit = submit.sort(['RankOrder'], ascending=True)
    
    # convert the probablities to rank order (required by the submission guidelines)
    for i in range(0, y_test_est.shape[0], 1):
        submit.iloc[i,1] = i + 1
    
    # re-sort by event ID
    submit = submit.sort(['EventId'], ascending=True)
    
    # convert the integer classification to (s, b)
    submit['Class'] = submit['Class'].map({1: 's', 0: 'b'})
    
    # force pandas to treat these columns at int (otherwise will write as floats)
    submit[['EventId', 'RankOrder']] = submit[['EventId', 'RankOrder']].astype(int)
    
    # finally create the submission file
    submit.to_csv('submission.csv', sep=',', index=False, index_label=False)


if __name__ == "__main__":
    
    import os, math, time, pickle, sys
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
    
    sys.path.append('/home/john/git/xgboost/wrapper')
    import xgboost as xgb
        
    # perform some initialization
    features = 31
    threshold = 85
    alg = 'xgboost' # bayes, logistic, boost, xgboost
    impute = 'none' # zeros, mean, none
    standardize = False
    whiten = False
    load_training_data = True
    load_model = False
    train_model = True
    save_model = False
    create_visualizations = False
    create_submission = False
    os.chdir('/home/john/git/kaggle/HiggsBosonChallenge')
    
    print 'Starting process...'
    print 'alg={0}, impute={1}, standardize={2}, whiten={3} threshold={4}'.format(
        alg, impute, standardize, whiten, threshold)
    
    if load_training_data == True:
        print 'Reading in training data...'
        training_data, X, y, w, scaler, pca = processTrainingData(
            '/home/john/data/training.csv', features, impute, standardize, whiten)
    
    if create_visualizations == True:
        print 'Creating visualizations...'
        visualize(training_data, X, y, scaler, pca)
    
    if load_model == True:
        print 'Loading model from disk...'
        model = load('model.pkl')
    
    if train_model == True: 
        print 'Training model on full data set...'
        model = train(X, y, w, alg, scaler, pca)
        
        print 'Calculating predictions...'
        y_prob, y_est = predict(X, model, alg, threshold, scaler, pca)
           
        print 'Calculating AMS...'
        ams = score(y, y_est, w)
        print'AMS =', ams
        
        print 'Performing cross-validation...'
        val = crossValidate(X, y, w, alg, scaler, pca, threshold)
        print'Cross-validation AMS =', val
    
    if save_model == True:
        print 'Saving model to disk...'
        save('model.pkl')
    
    if create_submission == True:
        print 'Reading in test data...'
        test_data, X_test = processTestData('/home/john/data/test.csv', features, impute)
        
        print 'Predicting test data...'
        y_test_prob, y_test_est = predict(X_test, model, alg, threshold, scaler, pca)
        
        print 'Creating submission file...'
        createSubmission(test_data, y_test_prob, y_test_est)
    
    print 'Process complete.'