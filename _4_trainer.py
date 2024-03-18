#############################################
#                                           #
#   This script handles the training for    #
#               all models                  #
#                                           #
#     It automatically runs each of the     #
#              models script                #
#                                           #
#############################################


import _4_preprocessing as preprocessing
import _4_trainer_config as trainer_config
import joblib
from nltk import RegexpTokenizer
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def return_text_counts(dataset):
    """
        Function returns a vector of elements consisting of tokens number of apparitions in a text
        input - dataset; Type Pandas DataFrame
        output - text_counts; Type CSR Matrix(Sparse Matrix that contain a lot of 0)
    """
    # Create tokenizer to extract only letters and digits
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # Configure vectorizer to ignore stop words and use the specific tokenizer
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    # Convert text to a CSR Matrix
    text_counts = vectorizer.fit_transform(dataset['text'].values.astype('U'))

    # Export vectorizer to disk
    joblib.dump(vectorizer, os.getcwd() + "\\Models\\Vectorizer.pkl")

    return text_counts


# Creates TF-IDF Matrix; equivalent to CountVectorizer + TfidfTransformer
def return_tfidf_vectorizer(dataset):
    """
        Function creates a TF-IDF Vectorizer -> a matrix of TF-IDF Features (equivalent to CountVectorizer + TfidfTransformer)
        input - dataset; Type Pandas DataFrame
        output - vectorizer; Type Matrix of TF-IDF features
    """
    # Create and configure vectorizer to control vocabulary size
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    # Convert text to a TF-IDF Matrix
    vectorizer.fit_transform(dataset['text'])

    return vectorizer


def run():
    """
        This function runs the training script for each model available
        input - None
        output - None
    """
    # Gets training data after it has been cleaned
    dataset = preprocessing.dataset_grabber(trainer_config.data_location, trainer_config.dataset_name,
                                            trainer_config.original_dataset_name)

    # Get matrix of TF-IDF Features
    vectorizer = return_tfidf_vectorizer(dataset)

    # Call each specific script for every model available to train
    for method in trainer_config.algorithms:
        for algorithm in trainer_config.algorithms[method]:
            module = __import__(method)
            function = getattr(module, algorithm)
            function(dataset, return_text_counts(dataset), vectorizer)
