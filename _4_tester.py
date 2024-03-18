#############################################
#                                           #
#   This script tests each model by giving  #
#       it input from terminal or API       #
#           and extracting output           #
#                                           #
#  It loads and runs each model with given  #
#                  input                    #
#                                           #
#############################################


import _3_twitter as twitter
import _3_reddit as reddit
import joblib
import keras.models
from keras.utils import pad_sequences
import numpy as np
import os


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)


def naive_bayes(BNB, vectorizer, text):
    """
        Function passes text through the Bernoulli Naive-Bayes Classifier Model
        input - BNB; Type Bernoulli Naive-Bayes Classifier
              - vectorizer; Type CSR Matrix(Sparse Matrix that contain a lot of 0)
              - text; Type STR
        output - sentiment review; Type STR
    """
    text = vectorizer.transform(text)
    return "Happy" if BNB.predict(text) == 4 else "Sad"


def support_vector_machine(SVM, optimized_vectorizer, text):
    """
        Function passes text through the Support Vector Machine Classifier Model
        input - SVM; Type Support Vector Machine Classifier
              - optimized_vectorizer; Type Matrix of TF-IDF features
              - text; Type STR
        output - sentiment review; Type STR
    """
    review_vector = optimized_vectorizer.transform(text)
    return "Happy" if SVM.predict(review_vector)[0] == 4 else "Sad"


def convolutional_neural_network(CNN, tokenizer, text):
    """
        Function passes text through the Convolutional Neural Network Model
        input - CNN; Type Convolutional Neural Network -> Keras Sequential
              - tokenizer; Type Keras Tokenizer Object
              - text; Type STR
        output - sentiment review; Type STR
    """
    sequence = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequence, 116)
    return "Happy" if np.argmax(CNN.predict(padded, verbose=0)) == 4 else "Sad"


def recurrent_neural_network(RNN, tokenizer, text):
    """
        Function passes text through the Recurrent Neural Network Model
        input - RNN; Type Recurrent Neural Network -> Keras Sequential
              - tokenizer; Type Keras Tokenizer Object
              - text; Type STR
        output - sentiment review; Type STR
    """
    sequence = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequence, 116)
    return "Happy" if np.argmax(RNN.predict(padded, verbose=0)) == 4 else "Sad"


def load_model(model):
    """
        Function loads a model from disk into a variable by using its name and path
        input - model; Type STR
        output - selected_model; Type Classifier or Sequential Neural Network
    """
    if model == "model_1_Naive-Bayes":
        selected_model = joblib.load(path[:len(path) - len(filename)] + '/Models/NaiveBayes/BernoulliNaiveBayes.pkl')
    elif model == "model_2_SVM":
        selected_model = joblib.load(
            path[:len(path) - len(filename)] + "/Models/SupportVectorMachine/SupportVectorMachine.pkl")
    elif model == "model_4_RNN":
        selected_model = keras.models.load_model(
            path[:len(path) - len(filename)] + "/Models/RecurrentNeuralNetwork/RecurrentNeuralNetwork.h5")
    else:
        selected_model = keras.models.load_model(
            path[:len(path) - len(filename)] + "/Models/ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h5")

    return selected_model


def run_model(model=None, query=None):
    """
        Function handles model load, text extraction from API and passing these parameters to other functions
        input - model; Type STR
              - query; Type STR
        output - Tuple of Lists; Type (STR, STR, STR(or INT in extreme cases))
    """

    # Load tokenizer or vectorizer depending on model
    if model == "model_3_CNN" or model == "model_4_RNN":
        tokenizer = joblib.load(path[:len(path) - len(filename)] + "/Models/Tokenizer.pkl")
    elif model == "model_1_Naive-Bayes":
        vectorizer = joblib.load(path[:len(path) - len(filename)] + '/Models/Vectorizer.pkl')
    elif model == "model_2_SVM":
        optimized_vectorizer = joblib.load(path[:len(path) - len(filename)] + '/Models/OptimizedVectorizer.pkl')
    else:
        tokenizer = joblib.load(path[:len(path) - len(filename)] + "/Models/Tokenizer.pkl")

    # Load selected model from disk
    selected_model = load_model(model)

    # Extract text and id from Reddit comments
    reddit_comments = reddit.retrieve_comments(query)
    comments_ids = reddit_comments[1]
    comments_texts = reddit_comments[0]

    review = []

    # Pass input text through the model for each block of text(usually comments or posts)
    for i in range(0, len(comments_texts)):
        if model == "model_1_Naive-Bayes":
            review.append(naive_bayes(selected_model, vectorizer, [comments_texts[i]]))
        elif model == "model_2_SVM":
            review.append(support_vector_machine(selected_model, optimized_vectorizer, [comments_texts[i]]))
        elif model == "model_3_CNN":
            review.append(convolutional_neural_network(selected_model, tokenizer, [comments_texts[i]]))
        elif model == "model_4_RNN":
            review.append(recurrent_neural_network(selected_model, tokenizer, [comments_texts[i]]))
        else:
            review.append(convolutional_neural_network(selected_model, tokenizer, [comments_texts[i]]))

    return review, comments_texts, comments_ids
