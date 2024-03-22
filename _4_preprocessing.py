#############################################
#                                           #
#   This script handles the preprocessing   #
#       of the data used for training       #
#                                           #
#      It can clean or export the text      #
#                                           #
#############################################


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from os.path import exists
import pandas as pd
import re
from string import punctuation


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)


def dataset_grabber(data_source, dataset_name, original_dataset_name):
    """
        Function will return the optimized dataset or create the file if it doesn't exist
        input - data_source -> path to file; Type STR
              - dataset_name; Type STR
              - original_dataset_name; Type STR
        output - dataset; Type Pandas DataFrame
    """

    if exists(dataset_name):
        # Extract wanted columns from dataset
        train_data = pd.read_csv(data_source, encoding="ISO-8859-1", header=None)
        train_data.columns = ['sentiment', 'text']
        train_data.fillna({"text": "NaN"}, inplace=True)  # Replace NaN with empty string
        train_data = train_data.tail(-1)
    else:
        # Read original file
        train_data = pd.read_csv(path[:len(path) - len(filename)] + original_dataset_name, encoding="ISO-8859-1", header=None)
        # Drop unused columns
        train_data.drop([1, 2, 3, 4], axis=1, inplace=True)
        # Clean text from first column
        train_data.iloc[:, 1] = train_data.iloc[:, 1].apply(text_cleaning)
        # Replace NaN with empty string to avoid errors
        train_data.fillna({"text": "NaN"}, inplace=True)
        # Replaces values in sentiment column to scale it for binary
        train_data.iloc[:, 1].replace(4, 1)
        # Create and export new file to disk
        train_data.to_csv(dataset_name, index=False)
        train_data.columns = ['sentiment', 'text']

    return train_data


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    """
        Function cleans the text from dataset of useless characters and tokens
        input - text; Type STR
              - remove_stop_words; Type BOOL
              - lemmatize_words; Type BOOL
        output - text -> cleaned; Type STR
    """

    # Set stop words to english
    stop_words = stopwords.words('english')

    # Remove links, special characters and float numbers from text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Remove stop words, default is True
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if w not in stop_words]
        text = " ".join(text)

    # Shorten words to their stems, default is True
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text
