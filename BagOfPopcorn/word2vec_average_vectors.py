import sys
sys.path.append('/home/git/kaggle/BagOfPopcorn/')

import logging
import nltk.data
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from old.Word2Vec.kaggle_utility import KaggleUtility


def make_feature_vec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph

    # Pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros(num_features, dtype='float32')

    nwords = 0

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords += 1
            feature_vec = np.add(feature_vec, model[word])

    # Divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0

    # Preallocate a 2D numpy array, for speed
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype='float32')

    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000 == 0:
            print 'Review %d of %d' % (counter, len(reviews))

        # Call the function (defined above) that makes average feature vectors
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)

        # Increment the counter
        counter += 1
    return review_feature_vecs


def get_clean_reviews(reviews):
    clean_reviews = []
    for review in reviews['review']:
        clean_reviews.append(KaggleUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


def main():
    data_dir = '/home/data/bag-of-popcorn/'

    # Read data from files
    train = pd.read_csv(data_dir + 'labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv(data_dir + 'testData.tsv', header=0, delimiter='\t', quoting=3)
    unlabeled_train = pd.read_csv(data_dir + 'unlabeledTrainData.tsv',  header=0,  delimiter='\t', quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print 'Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n' % \
          (train['review'].size, test['review'].size, unlabeled_train['review'].size)

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Split the labeled and unlabeled training sets into clean sentences

    # Initialize an empty list of sentences
    sentences = []

    print 'Parsing sentences from training set'
    for review in train['review']:
        sentences += KaggleUtility.review_to_sentences(review, tokenizer)

    print 'Parsing sentences from unlabeled set'
    for review in unlabeled_train['review']:
        sentences += KaggleUtility.review_to_sentences(review, tokenizer)

    # Set parameters and train the word2vec model

    # Import the built-in logging module and configure it so that BagOfPopcorn
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print 'Training BagOfPopcorn model...'
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using BagOfPopcorn.load()
    model_name = '300features_40minwords_10context'
    model.save(data_dir + model_name)

    model.doesnt_match('man woman child kitchen'.split())
    model.doesnt_match('france england germany berlin'.split())
    model.doesnt_match('paris berlin london austria'.split())
    model.most_similar('man')
    model.most_similar('queen')
    model.most_similar('awful')

    # Create average vectors for the training and test sets

    print 'Creating average feature vecs for training reviews'

    train_data_vecs = get_avg_feature_vecs(get_clean_reviews(train), model, num_features)

    print 'Creating average feature vecs for test reviews'

    test_data_vecs = get_avg_feature_vecs(get_clean_reviews(test), model, num_features)

    # Fit a random forest to the training set, then make predictions

    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print 'Fitting a random forest to labeled training data...'
    forest = forest.fit(train_data_vecs, train['sentiment'])

    # Test & extract results
    result = forest.predict(test_data_vecs)

    # Write the test results
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
    output.to_csv(data_dir + 'Word2Vec_AverageVectors.csv', index=False, quoting=3)
    print 'Wrote Word2Vec_AverageVectors.csv'


if __name__ == "__main__":
    main()