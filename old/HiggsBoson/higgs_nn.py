import os, math
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing
from theano import function
from pylearn2.config import yaml_parse
from pylearn2.utils import serial


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


def create_nn_pre_train_file(original_filename, new_filename, impute, scaler, pca):
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
    
    if scaler is not None:
        X = scaler.transform(X)
    
    if pca is not None:
        X = pca.transform(X)
    
    combined_data = pd.DataFrame(X, columns=combined_data.columns.values)
    combined_data.to_csv(new_filename, sep=',', index=False)


def create_nn_training_file(training_data, features, impute, scaler, pca, filename):
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
    
    X = nn_training_data.iloc[:, 1:features+1].values
    
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999)
        X = imp.fit_transform(X)
    elif impute == 'zeros':
        X[X == -999] = 0
    
    if scaler is not None:
        X = scaler.transform(X)
    
    if pca is not None:
        X = pca.transform(X)
    
    X = np.insert(X, 0, nn_training_data['NN_Label'].values, 1)
    
    nn_training_data = pd.DataFrame(X, columns=nn_training_data.columns.values)
    nn_training_data.to_csv(filename, sep=',', index=False)


def train(model_definition_file, data_dir):
    """
    Trains a neural network model using the pylearn2 library.
    """
    with open(model_definition_file, 'r') as f:
        train_nn = f.read()
    
    hyper_params = {'data_dir': data_dir,
                    'num_features': 30,
                    'dim_h0': 50,
                    'batch_size': 100,
                    'max_epochs': 10,
                    'train_start': 0,
                    'train_stop': 150000,
                    'valid_start': 150001,
                    'valid_stop': 200000,
                    'test_start': 200001,
                    'test_stop': 250000}
    train_nn = train_nn % hyper_params
    train_nn = yaml_parse.load(train_nn)
    train_nn.main_loop()


def predict(X, threshold, scaler, pca, model_file):
    """
    Compiles a Theano function using the pylearn 2 model's fprop
    to predict the probability of a positive outcome, and converts
    to a binary prediction based on the cutoff percentage.
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    if pca is not None:
        X = pca.transform(X)
    
    # Load the model
    model = serial.load(model_file)
    
    # Create Theano function to compute probability
    x = model.get_input_space().make_theano_batch()
    y = model.fprop(x)
    pred = function([x], y)
    
    # Convert to a prediction
    y_prob = pred(X)[:, 1]
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
    impute = 'zeros'  # zeros, mean, none
    standardize = True
    whiten = False
    load_training_data = True
    train_model = True
    create_nn_files = True
    train_nn_model = True
    create_submission_file = False
    code_dir = '/home/john/git/kaggle/HiggsBoson/'
    data_dir = '/home/john/data/higgs-boson/'
    pretrain_file = 'combined.csv'
    training_file = 'training.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    pretrain_nn_file = 'combined_nn.csv'
    training_nn_file = 'training_nn.csv'
    model_definition_file = 'mlp.yaml'
    model_file = 'mlp.pkl'
    
    os.chdir(code_dir)
    
    print 'Starting process...'
    print 'impute={0}, standardize={1}, whiten={2} threshold={3}'.format(
        impute, standardize, whiten, threshold)
    
    if load_training_data:
        print 'Reading in training data...'
        training_data, X, y, w, scaler, pca = process_training_data(
            data_dir + training_file, features, impute, standardize, whiten)
    
    if train_model:
        print 'Running neural network process...'
        
        if create_nn_files:
            print 'Creating training files...'
            create_nn_training_file(training_data, features, impute, scaler, pca,
                data_dir + training_nn_file)
            create_nn_pre_train_file(data_dir + pretrain_file,
                data_dir + pretrain_nn_file, impute, scaler, pca)
        
        if train_nn_model:
            print 'Training the model...'
            train(code_dir + model_definition_file, data_dir)
        
        print 'Calculating predictions...'
        y_prob, y_est = predict(X, threshold, scaler, pca, data_dir + model_file)
        
        print 'Calculating AMS...'
        ams_val = score(y, y_est, w)
        print 'AMS =', ams_val
    
    if create_submission_file:
        print 'Reading in test data...'
        test_data, X_test = process_test_data(data_dir + test_file, features, impute)
        
        print 'Predicting test data...'
        y_test_prob, y_test_est = predict(X_test, threshold, scaler, pca, data_dir + model_file)
        
        print 'Creating submission file...'
        create_submission(test_data, y_test_prob, y_test_est, data_dir + submit_file)
    
    print 'Process complete.'


if __name__ == "__main__":
    main()