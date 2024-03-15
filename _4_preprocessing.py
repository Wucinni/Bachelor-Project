#################################################
#                                               #
# Author: Geanaliu Andy Dennis                  #
#                                               #
#################################################


import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from os.path import exists

x = 0

def dataset_grabber(data_source):
    if exists('ToBeWrittenNameOfCSV.csv'):
        train_data = pd.read_csv(r'C:\Users\Dennis\PycharmProjects\Bachelor-Project\_1_1600000tweets.csv', encoding="ISO-8859-1", header=None)
        train_data.columns = ['sentiment', 'text']
        train_data = train_data.tail(-1)
    else:
        train_data = pd.read_csv(data_source, encoding="ISO-8859-1", header=None)
        train_data.drop([1, 2, 3, 4], axis=1, inplace=True)
        #train_data.iloc[:, 1] = train_data.iloc[:, 1].apply(text_cleaning)
        train_data.iloc[:, 1].replace(4, 1)
        train_data.to_csv('_1_1600000tweetsNOTCLEANED.csv', index=False)
        train_data.columns = ['sentiment', 'text']

    return train_data

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    stop_words = stopwords.words('english')
    global x
    if x % 10000 == 0:
        print("In Cleaning", x, "/ 1599999")
    x = x + 1

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if w not in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text

