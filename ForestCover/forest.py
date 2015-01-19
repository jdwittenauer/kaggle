# -*- coding: utf-8 -*-
"""
@author: John Wittenauer

@notes: This script was tested on 64-bit Windows 7 using the Anaconda 2.0
distribution of 64-bit Python 2.7.
"""

import os


def main():
    # perform some initialization
    load_training_data = True
    load_model = False
    train_model = False
    save_model = False
    create_visualizations = True
    create_submission_file = False
    code_dir = 'C:\\Users\\John\\PycharmProjects\\Kaggle\\ForestCover\\'
    data_dir = 'C:\\Users\\John\\Documents\\Kaggle\\ForestCover\\'
    training_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submission.csv'
    model_file = 'model.pkl'

    os.chdir(code_dir)

    print 'Starting process...'

    if load_training_data:
        print 'Reading in training data...'
        # TODO

    if create_visualizations:
        print 'Creating visualizations...'
        # TODO

    if load_model:
        print 'Loading model from disk...'
        # TODO

    if train_model:
        print 'Training model on full data set...'
        # TODO

        print 'Calculating predictions...'
        # TODO

        print 'Calculating score...'
        # TODO

        print 'Performing cross-validation...'
        # TODO

    if save_model:
        print 'Saving model to disk...'
        # TODO

    if create_submission_file:
        print 'Reading in test data...'
        # TODO

        print 'Predicting test data...'
        # TODO

        print 'Creating submission file...'
        # TODO

    print 'Process complete.'


if __name__ == "__main__":
    main()