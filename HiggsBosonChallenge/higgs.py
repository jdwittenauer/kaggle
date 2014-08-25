# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 14:43:59 2014

@author: John Wittenauer
"""

import os,random,string,math,csv,pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl

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

# set current directory
os.chdir("C:\Users\John\Documents\Github\kaggle\HiggsBosonChallenge")

# read in training data
print 'Reading in training data...'
trainingData=pd.read_csv('training.csv', sep=',')

# add nominal column with the label
temp = trainingData['Label'].replace(to_replace=['s','b'], value=[1,0])
trainingData['Nominal'] = temp

# create numpy arrays for model training and cross-validation
X = trainingData.iloc[:,1:31].values
y = trainingData.iloc[:,33].values
w = trainingData.iloc[:,31].values

# load previously trained model if applicable
#model_file = open('model.pkl', 'rb')
#model = pickle.load(model_file)
#model_file.close()

# train a new model if needed
print 'Training model...'
model = skl.linear_model.LogisticRegression()
model.fit(X, y)

# persist the model if desired
#model_file = open('model.pkl', 'wb')
#pickle.dump(model, model_file)
#model_file.close()

# make a prediction on the training data using the model
y_est = model.predict(X)

# the model also calculates a probability
y_est_probability = model.predict_proba(X)[:,1]

# choose a new threshold and apply that to the classification
threshold = np.percentile(y_est_probability, 85)
y_est = y_est_probability > threshold

# create weighted signal and background sets and calculate the AMS
print 'Calculating AMS...'
y_est_signal = w * (y == 1.0)
y_est_background = w * (y == 0.0)
s = np.sum(y_est_signal * (y_est == 1.0))
b = np.sum(y_est_background * (y_est == 1.0))
score = AMS(s, b)
print'AMS =', score

# now read in the test data
print 'Reading in test data...'
testData = pd.read_csv('test.csv', sep=',')

# create test set and use the model to score the data
X_test = testData.iloc[:,1:31].values
y_test = model.predict_proba(X_test)[:,1]

# create a new datafrane with the submission data
print 'Creating submission file...'
temp = pd.DataFrame(y_test, columns=['RankOrder'])
y_test_est = y_test > threshold
temp2 = pd.DataFrame(y_test_est, columns=['Class'])
result = pd.DataFrame([testData.EventId, temp.RankOrder, temp2.Class]).transpose()

# sort it so they're in the ascending order by probability
result = result.sort(['RankOrder'], ascending=True)

# convert the probablities to rank order (required by the submission guidelines)
for i in range(0, X_test.shape[0], 1):
    result.iloc[i,1] = i + 1

# re-sort by event ID
result = result.sort(['EventId'], ascending=True)

# convert the integer classification to (s, b)
result['Class'] = result['Class'].map({1: 's', 0: 'b'})

# force pandas to treat these columns at int (otherwise will write as floats)
result[['EventId', 'RankOrder']] = result[['EventId', 'RankOrder']].astype(int)

# finally create the submission file
result.to_csv('submission.csv', sep=',', index=False, index_label=False)

print 'Process complete.'