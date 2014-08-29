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


def loadModel(filename):
    """
    Load a previously training model from disk.
    """   
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model


def saveModel(filename):
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
    
    # optionally impute the -999 values to 0
    if impute == True:
        for i in training_data.columns:
            col = training_data[i]
            col[col.isin([-999])] = 0
    
    X = training_data.iloc[:,1:features].values
    y = training_data.iloc[:,features+2].values
    w = training_data.iloc[:,features].values
    
    # create a standardization transform
    scaler = preprocessing.StandardScaler()
    if standardize == True:
        scaler.fit(X)
        X = scaler.transform(X)
    
    return X, y, w, scaler


def trainModel(X, y, alg):
    """
    Trains a new model using the training data.
    """
    t0 = time.time()    
    
    # 1.73 impute=True standardize=True threshold=85
    if alg == 'logistic':
        model = linear_model.LogisticRegression()
    # 0.99 impute=False standardize=True threshold=85 
    elif alg == 'bayes':
        model = naive_bayes.GaussianNB()
    # 3.48 impute=False standardize=True threshold=85
    elif alg == 'boost':
        model = ensemble.GradientBoostingClassifier(n_estimators=50, 
            max_depth=5, min_samples_leaf=200, max_features=10, verbose=1)
            
    model.fit(X, y)
    
    t1 = time.time()
    print 'Training took %0.3f s.' % (t1 - t0)
    
    return model


def predict(X, model, threshold):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    y_prob = model.predict_proba(X)[:,1]
    cutoff = np.percentile(y_prob, threshold)
    y_est = y_prob > cutoff

    return y_est


def scoreModel(w, y, y_est):
    """
    Create weighted signal and background sets and calculate the AMS.
    """
    y_signal = w * (y == 1.0)
    y_background = w * (y == 0.0)
    s = np.sum(y_signal * (y_est == 1.0))
    b = np.sum(y_background * (y_est == 1.0))
    
    return AMS(s, b)


def processTestData(fileName, features, impute, standardize, scaler):
    """
    Reads in test data and prepares numpy arrays.
    """
    test_data = pd.read_csv(fileName, sep=',')
    
    # optionally impute the -999 values to 0
    if impute == True:
        for i in test_data.columns:
            col = test_data[i]
            col[col.isin([-999])] = 0
    
    X_test = test_data.iloc[:,1:features].values
    
    # standardize using the same transform from training
    if standardize == True:
        X_test = scaler.transform(X_test)
    
    return test_data, X_test


def predictTest(X_test, model, threshold):
    """
    Create test set and use the model to score the data.
    """
    y_test_prob = model.predict_proba(X_test)[:,1] 
    cutoff = np.percentile(y_test_prob, threshold)
    y_test_est = y_test_prob > cutoff
    
    return y_test_prob, y_test_est


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
    
    import os,random,string,math,time,csv,pickle
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import ensemble
    from sklearn import linear_model
    from sklearn import naive_bayes
    from sklearn import neighbors
    from sklearn import preprocessing
    from sklearn import svm
        
    # perform some initialization
    features = 31
    threshold = 85
    impute = True
    standardize = True
    load_model = False
    save_model = False
    create_submission = False
    alg = 'svm'
    os.chdir("C:\Users\John\Documents\Kaggle\Higgs Boson Challenge")
    
    print 'Reading in training data...'
    X, y, w, scaler = processTrainingData('training.csv', features, impute, standardize)
    
    if load_model == True:
        print 'Loading model from disk...'
        model = loadModel('model.pkl')
    
    print 'Training model...'
    model = trainModel(X, y, alg)
    
    if save_model == True:
        print 'Saving model to disk...'
        saveModel('model.pkl')
    
    print 'Calculating predictions...'
    y_est = predict(X, model, threshold)
       
    print 'Calculating AMS...'
    score = scoreModel(w, y, y_est)
    print'AMS =', score
    
    if create_submission == True:
        print 'Reading in test data...'
        test_data, X_test = processTestData('test.csv', features, impute, standardize, scaler)
        
        print 'Predicting test data...'
        y_test_prob, y_test_est = predictTest(X_test, model, threshold)
        
        print 'Creating submission file...'
        createSubmission(test_data, y_test_prob, y_test_est)
    
    print 'Process complete.'