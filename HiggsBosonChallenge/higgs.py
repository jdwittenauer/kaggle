# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 14:43:59 2014

@author: John Wittenauer
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


def processTrainingData(fileName, features, impute, standardize):
    """
    Reads in training data and prepares numpy arrays.
    """
    training_data = pd.read_csv('training.csv', sep=',')
    
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
    scaler = preprocessing.StandardScaler()
    if standardize == True:
        scaler.fit(X)
    
    return X, y, w, scaler


def train(X, y, alg, standardize, scaler):
    """
    Trains a new model using the training data.
    """
    t0 = time.time()    
    
    # 0.99/0.99 impute=none standardize=True threshold=85
    if alg == 'bayes':
        model = naive_bayes.GaussianNB()
    # 1.74/1.74 impute=zeros standardize=True threshold=85
    elif alg == 'logistic':
        model = linear_model.LogisticRegression()
    # 3.41/3.36 impute=none standardize=True threshold=85
    elif alg == 'boost':
        model = ensemble.GradientBoostingClassifier(learning_rate=0.1,
            n_estimators=50, max_depth=5, min_samples_split=2,
            min_samples_leaf=200, subsample=1.0, max_features=10)
    else:
        print 'No model defined for ' + alg
        exit()
    
    if standardize == True:
        X = scaler.transform(X)
        
    model.fit(X, y)
    
    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)
    
    return model


def predict(X, model, threshold, standardize, scaler):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    if standardize == True:
        X = scaler.transform(X)
    
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


def crossValidate(X, y, alg, standardize, scaler, w, threshold):
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
        model = train(X_train, y_train, alg, standardize, scaler)
        
        # predict and score preformance on the validation set
        y_val_prob, y_val_est = predict(X_val, model, threshold, standardize, scaler)
        scores[i] = score(y_val, y_val_est, w_val)
        i = i + 1
    
    return np.mean(scores)


def processTestData(fileName, features, impute):
    """
    Reads in test data and prepares numpy arrays.
    """
    test_data = pd.read_csv(fileName, sep=',')
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
    
    import os, math, time, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import cross_validation
    from sklearn import ensemble
    from sklearn import linear_model
    from sklearn import naive_bayes
    from sklearn import preprocessing
        
    # perform some initialization
    features = 31
    threshold = 85
    alg = 'boost' # bayes, logistic, boost
    impute = 'none' # zeros, mean, none
    standardize = True
    load_model = False
    save_model = False
    create_submission = False
    os.chdir("C:\Users\John\Documents\Kaggle\Higgs Boson Challenge")
    
    print 'Starting process...'
    print 'alg={0}, impute={1}, standardize={2}, threshold={3}'.format(
        alg, impute, standardize, threshold)
    print 'Reading in training data...'
    X, y, w, scaler = processTrainingData('training.csv', features, impute, standardize)
    
    if load_model == True:
        print 'Loading model from disk...'
        model = load('model.pkl')
    
    print 'Training model on full data set...'
    model = train(X, y, alg, standardize, scaler)
    
    print 'Calculating predictions...'
    y_prob, y_est = predict(X, model, threshold, standardize, scaler)
       
    print 'Calculating AMS...'
    ams = score(y, y_est, w)
    print'AMS =', ams
    
    print 'Performing cross-validation...'
    val = crossValidate(X, y, alg, standardize, scaler, w, threshold)
    print'Cross-validation AMS =', val
    
    if save_model == True:
        print 'Saving model to disk...'
        save('model.pkl')
    
    if create_submission == True:
        print 'Reading in test data...'
        test_data, X_test = processTestData('test.csv', features, impute)
        
        print 'Predicting test data...'
        y_test_prob, y_test_est = predict(X_test, model, threshold, standardize, scaler)
        
        print 'Creating submission file...'
        createSubmission(test_data, y_test_prob, y_test_est)
    
    print 'Process complete.'