# -*- coding: utf-8 -*-
"""
@author: John Wittenauer

@notes: This script was tested on 64-bit Ububtu 14 using the Anaconda 2.0
distribution of 64-bit Python 2.7.
"""

import os, math
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing
from theano import function
from pylearn2.config import yaml_parse
from pylearn2.utils import serial


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


def createNNPreTrainFile(original_filename, new_filename, impute, scaler, pca):
    """
    Creates a non-labeled data set with transforms applied to be used
    by pylearn2's csv data set class.
    """
    combined_data = pd.read_csv(original_filename, sep=',')
    
    X = combined_data.values
    
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X = imp.fit_transform(X)
    elif impute == 'zeros':
        X[X == -999] = 0
    
    if scaler != None:
        X = scaler.transform(X)
    
    if pca != None:
        X = pca.transform(X)
    
    combined_data = pd.DataFrame(X, columns=combined_data.columns.values)
    combined_data.to_csv(new_filename, sep=',', index=False)


def createNNTrainingFile(training_data, features, impute, scaler, pca, filename):
    """
    Creates a labeled training set with transforms applied to be used
    by pylearn2's csv data set class.
    """
    nn_training_data = training_data
    
    nn_training_data.insert(0, 'NN_Label', nn_training_data['Nominal'].values)
    
    nn_training_data.drop('EventId', axis=1, inplace=True)
    nn_training_data.drop('Weight', axis=1, inplace=True)
    nn_training_data.drop('Label', axis=1, inplace=True)
    nn_training_data.drop('Nominal', axis=1, inplace=True)
    
    X = nn_training_data.iloc[:,1:features].values
    
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X = imp.fit_transform(X)
    elif impute == 'zeros':
        X[X == -999] = 0
    
    if scaler != None:
        X = scaler.transform(X)
    
    if pca != None:
        X = pca.transform(X)
    
    X = np.insert(X, 0, nn_training_data['NN_Label'].values, 1)
    
    nn_training_data = pd.DataFrame(X, columns=nn_training_data.columns.values)
    nn_training_data.to_csv(filename, sep=',', index=False)


def train(yaml):
    """
    Trains a neural network model using the pylearn2 library.
    """
    with open(yaml, 'r') as f:
        train_nn = f.read()
    
    hyper_params = { 'batch_size' : 100 }
    train_nn = train_nn % (hyper_params)
    train_nn = yaml_parse.load(train_nn)
    train_nn.main_loop()


def predict(X, threshold, scaler, pca, model_file):
    """
    Compiles a Theano function using the pylearn 2 model's fprop
    to predict the probability of a positive outcome, and converts
    to a binary prediction based on the cutoff percentage.
    """
    if scaler != None:
        X = scaler.transform(X)
    
    if pca != None:
        X = pca.transform(X)
    
    # Load the model
    model = serial.load(model_file)
    
    # Create Theano function to compute probability
    x = model.get_input_space().make_theano_batch()
    y = model.fprop(x)
    pred = function([x], y)
    
    # Convert to a prediction
    y_prob = pred(X)[:,1]
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
        X_test[X_test == -999] = 0
    
    return test_data, X_test


def createSubmission(test_data, y_test_prob, y_test_est, data_dir):
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
    submit.to_csv(data_dir + '/submission.csv', sep=',', index=False, index_label=False)


def main():
    # perform some initialization
    features = 31
    threshold = 85
    impute = 'none' # zeros, mean, none
    standardize = False
    whiten = False
    load_training_data = True
    train_model = True
    create_nn_files = False
    train_nn_model = True
    create_submission = False
    model_train_file = 'softmax.yaml'
    model_file = 'softmax.pkl'
    code_dir = '/home/john/git/kaggle/HiggsBosonChallenge'
    data_dir = '/home/john/data'
    
    os.chdir(code_dir)
    
    print 'Starting process...'
    print 'impute={1}, standardize={2}, whiten={3} threshold={4}'.format(
        impute, standardize, whiten, threshold)
    
    if load_training_data == True:
        print 'Reading in training data...'
        training_data, X, y, w, scaler, pca = processTrainingData(
            data_dir + '/training.csv', features, impute, standardize, whiten)
    
    if train_model == True:
        print 'Running neural network process...'
        
        if create_nn_files == True:
            print 'Creating training files...'
            createNNTrainingFile(training_data, features, impute, scaler, pca,
                data_dir + '/training_nn.csv')
            createNNPreTrainFile(data_dir + '/combined.csv', 
                data_dir + '/combined_nn.csv', impute, scaler, pca)
        
        if train_nn_model == True:
            print 'Training the model...'
            train(model_train_file)
        
        print 'Calculating predictions...'
        y_prob, y_est = predict(X, threshold, scaler, pca, model_file)
        
        print 'Calculating AMS...'
        ams = score(y, y_est, w)
        print'AMS =', ams
    
    if create_submission == True:
        print 'Reading in test data...'
        test_data, X_test = processTestData(data_dir + '/test.csv', features, impute)
        
        print 'Predicting test data...'
        y_test_prob, y_test_est = predict(X_test, threshold, scaler, pca, model_file)
        
        print 'Creating submission file...'
        createSubmission(test_data, y_test_prob, y_test_est, data_dir)
    
    print 'Process complete.'


if __name__ == "__main__":
    main()