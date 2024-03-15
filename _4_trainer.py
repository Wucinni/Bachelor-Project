#############################################
#                                           #
#   This script grabs training data         #
#   and runs each model by script name      #
#                                           #
#############################################

import joblib
from nltk import RegexpTokenizer
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import _4_preprocessing as preprocessing
import _4_trainer_config as trainer_config

# Creates token count matrix
def return_text_counts(dataset):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = vectorizer.fit_transform(dataset['text'].values.astype('U'))

    joblib.dump(vectorizer, os.getcwd() + "\\Models\\Vectorizer.pkl")

    return text_counts


# Creates TF-IDF Matrix; equivalent to CountVectorizer + TfidfTransformer
def return_tfidf_vectorizer(dataset):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    vectorizer.fit_transform(dataset['text'])

    return vectorizer


print(type(return_tfidf_vectorizer(preprocessing.dataset_grabber(trainer_config.data_location))))


def run():
    # Gets training data after it has been cleaned
    dataset = preprocessing.dataset_grabber(trainer_config.data_location)

    vectorizer = return_tfidf_vectorizer(dataset)
    algorithm_info = []

    # Following code calls specific script for every model trained
    for method in trainer_config.algorithms:
        for algorithm in trainer_config.algorithms[method]:
            module = __import__(method)
            function = getattr(module, algorithm)
            algorithm_info.append(function(dataset, return_text_counts(dataset), vectorizer))

    return algorithm_info
