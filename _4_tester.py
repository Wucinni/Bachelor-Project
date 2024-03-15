import keras.models
import numpy as np
from keras.utils import pad_sequences
import joblib
import _3_twitter as twitter
import _3_reddit as reddit


def naive_bayes(BNB, vectorizer, text):
    text = vectorizer.transform(text)
    Y = (BNB.predict(text))

    if Y[0] == 4:
        print("Naive-Bayes: Happy")
        return "Happy"
    else:
        print("Naive-Bayes: Sad")
        return "Sad"


def support_vector_machine(SVM, optimized_vectorizer, text):
    review_vector = optimized_vectorizer.transform(text)
    if SVM.predict(review_vector)[0] == 4:
        print("SVM: Happy")
        return "Happy"
    else:
        print("SVM: Sad")
        return "Sad"


def convolutional_neural_network(CNN, tokenizer, text):
    #tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequence, 116)

    review = CNN.predict(padded, verbose=0)
    prediction = np.argmax(review)
    if (prediction == 4):
        print(f"CNN: Happy")
        return "Happy"
    else:
        print(f"CNN: Sad")
        return "Sad"


def recurrent_neural_network(RNN, tokenizer, text):
    #tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequence, 116)

    prediction = RNN.predict(padded, verbose=0)
    if np.argmax(prediction) == 4:
        print("RNN: Happy")
        return "Happy"
    elif np.argmax(prediction) == 0:
        print("RNN: Sad")
        return "Sad"


def load_model(model):
    if model == "model_1_Naive-Bayes":
        selected_model = joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/NaiveBayes/BernoulliNaiveBayes.pkl')
    elif model == "model_2_SVM":
        selected_model = joblib.load(
            "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/SupportVectorMachine/SupportVectorMachine.pkl")
    elif model == "model_4_RNN":
        selected_model = keras.models.load_model(
            "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/RecurrentNeuralNetwork/RecurrentNeuralNetwork.h5")
    else:
        selected_model = keras.models.load_model(
            "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h5")
    return selected_model


def loader(model, query):
    print("Started Loading tokenizer")
    if model == "model_3_CNN" or model == "model_4_RNN":
        tokenizer = joblib.load("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")
    elif model == "model_1_Naive-Bayes":
        vectorizer = joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Vectorizer.pkl')
    elif model == "model_2_SVM":
        optimized_vectorizer =  joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/OptimizedVectorizer.pkl')
    else:
        tokenizer = joblib.load("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")


    print("Started Loading Model")
    selected_model = load_model(model)

    print("Getting Reddit Coments")
    all = reddit.retrieve_comments(query)
    print("Have All Reddit Comments")
    comments_ids = all[1]
    text = all[0]

    review = []

    for i in range(0, len(text)):
        if model == "model_1_Naive-Bayes":
            review.append(naive_bayes(selected_model, vectorizer, [text[i]]))
        elif model == "model_2_SVM":
            review.append(support_vector_machine(selected_model, optimized_vectorizer, [text[i]]))
        elif model == "model_3_CNN":
            review.append(convolutional_neural_network(selected_model, tokenizer, [text[i]]))
        elif model == "model_4_RNN":
            review.append(recurrent_neural_network(selected_model, tokenizer, [text[i]]))
        else:
            review.append(convolutional_neural_network(selected_model, tokenizer, [text[i]]))

    print("Texts:", text)
    print("REVIEW:", review)

    return review, text, comments_ids
