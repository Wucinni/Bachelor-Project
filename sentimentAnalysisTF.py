#################################################
#                                               #
# This version utilizes Tensorflow for training #
#                                               #
#################################################

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras import layers
from keras import losses
# To include in further updates
# from keras.models import load_model

# Data manipulation libraries
import numpy as np
import pandas as pd

# Nltk Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
for dependency in (
        "brown",
        "names",
        "wordnet",
        "averaged_perceptron_tagger",
        "universal_tagset",
        "stopwords"
):
    nltk.download(dependency)

# Language Libraries
from string import punctuation
import re  # regular expression

# General Use Libraries
import datetime
import warnings


print("Version: ", tf.__version__)
warnings.filterwarnings("ignore")


# Seeding
np.random.seed(123)


# First part: Data Loading
# To test encoding for UTF-8 as there were errors initially
train_data = pd.read_csv("training.1600000.processed.noemoticon.csv", on_bad_lines='skip', encoding='latin-1')


# TODO: To implement Logger
print("Training data finished loading.")


# Second part: Cleaning Text
stop_words = stopwords.words('english')


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
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


# Using text_cleaning function on available data
train_data["text"] = train_data["text"].apply(text_cleaning)


# TODO: To implement Logger
print("Data cleaned.")


# Cutting useless data from csv & saving
train_data = train_data.drop(train_data.columns[[1, 2, 3, 4]], axis=1)
# Replacing sentiment value from 4 with 1 to work with tf
train_data['sentiment'] = train_data['sentiment'].replace(4, 1)
train_data.to_csv('training.1600000.processed.no-emoticon_v2.csv', index=False)


# TODO: To implement Logger
print("New data saved.")


train_dataset = tf.data.experimental.make_csv_dataset(
    "training.1600000.processed.no-emoticon_v2.csv",
    batch_size=1,
    column_names=['text', 'sentiment'],
    label_name="sentiment",
    num_epochs=1
)

# TODO: To implement Logger
print("Loaded CSV[new data]!")


# To be deleted, used only for data validation
train_examples_batch, train_labels_batch = next(iter(train_dataset.batch(10)))

# TODO: To move this dataset in alternative analyzer(Movie reviews)
train_ds, val_ds, test_ds = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


def preprocess(x):
    return text_vectorizer(x)


text_vectorizer = layers.TextVectorization(
     output_mode='multi_hot',
     # Vocabulary size, I'll default it to 10k
     max_tokens=10000
     )


# Map text for vectorizer
features = train_dataset.map(lambda y, x: x)


# Optional: only to set internal state from input data
text_vectorizer.adapt(features)


# Defining the model
inputs = keras.Input(shape=(1,), dtype='string')
outputs = layers.Dense(1)(preprocess(inputs))
model = keras.Model(inputs, outputs)


# Compiling the model
model.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
model.summary()


# Create TensorBoard folders
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# Create callbacks
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
]


# Number of epochs to teach model, I'll default it to 25 for now
epochs = 25


history = model.fit(
    train_ds.shuffle(buffer_size=10000).batch(512),
    epochs=epochs,
    validation_data=val_ds.batch(512),
    callbacks=my_callbacks,
    verbose=1)


# Saving final model
model.save('twitterNLP_Model')
