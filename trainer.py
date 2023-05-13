import joblib
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import preprocessing
import trainer_config


def return_text_counts(dataset):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = vectorizer.fit_transform(dataset['text'].values.astype('U'))

    joblib.dump(vectorizer, "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Vectorizer.pkl")

    return text_counts


def return_tfidf_vectorizer(dataset):
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)

    vectorizer.fit_transform(dataset['text'])

    return vectorizer


def run():
    dataset = preprocessing.dataset_grabber(trainer_config.data_location)

    vectorizer = return_tfidf_vectorizer(dataset)

    algorithmInfo = []

    for method in trainer_config.algorithms:
        for algorithm in trainer_config.algorithms[method]:
            module = __import__(method)
            function = getattr(module, algorithm)
            algorithmInfo.append(function(dataset, return_text_counts(dataset), vectorizer))

    return algorithmInfo




