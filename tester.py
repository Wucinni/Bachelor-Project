import keras.models
import numpy as np
from keras.utils import pad_sequences
import joblib
import twitter


def naive_bayes(MNB, CNB, BNB, vectorizer, text):
    text = vectorizer.transform(text)

    X = (MNB.predict(text))
    Y = (BNB.predict(text))
    Z = (CNB.predict(text))

    output = []
    if X[0] == 4:
        print("Multinomial Happy")
        output.append("Naive Bayes Multinomial Happy")
    else:
        print("Multinomial: Sad")
        output.append("Naive Bayes Multinomial: Sad")

    if Y[0] == 4:
        print("Bernoulli: Happy")
        output.append("Naive Bayes Bernoulli: Happy")
    else:
        print("Bernoulli: Sad")
        output.append("Naive Bayes Bernoulli: Sad")

    if Z[0] == 4:
        print("Complement: Happy")
        output.append("Naive Bayes Complement: Happy")
    else:
        print("Complement: Sad")
        output.append("Naive Bayes Complement: Sad")

    return output


def support_vector_machine(SVM, optimized_vectorizer, text):
    review_vector = optimized_vectorizer.transform(text)
    if SVM.predict(review_vector)[0] == 4:
        print("SVM: Happy")
        return "SVM: Happy"
    else:
        print("SVM: Sad")
        return "SVM: Sad"


def convolutional_neural_network(CNN, tokenizer, text):
        #tokenizer.fit_on_texts(text)
        sequence = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(sequence, 116)

        review = CNN.predict(padded, verbose = 0)
        prediction = np.argmax(review) # axis=1
        if (prediction == 4):
            print(f"CNN: Happy")
            return f"CNN: Happy"
        else:
            print(f"CNN: Sad")
            return f"CNN: Sad"

def recurrent_neural_network(RNN, tokenizer, text):
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequence, maxlen=200, dtype='int32', value=0)
    prediction = RNN.predict(padded, batch_size=1, verbose=0)
    if (np.argmax(prediction) == 0):
        print("RNN: Happy")
        return "RNN: Happy"
    elif (np.argmax(prediction) == 1):
        print("RNN: Sad")
        return "RNN: Sad"


def loader():
    tokenizer = joblib.load("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")
    vectorizer = joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Vectorizer.pkl')
    optimized_vectorizer =  joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/OptimizedVectorizer.pkl')


    MNB = joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/NaiveBayes/MultinomialNaiveBayes.pkl')
    CNB = joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/NaiveBayes/ComplementNaiveBayes.pkl')
    BNB = joblib.load('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/NaiveBayes/BernoulliNaiveBayes.pkl')

    SVM = joblib.load("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/SupportVectorMachine/SupportVectorMachine.pkl")

    CNN = keras.models.load_model("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h5")

    RNN = keras.models.load_model("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/LongShortTermMemory/LongShortTermMemory.h5")


    # while True:
    #     text = [input("Enter Phrase:")]
    #     #naive_bayes(MNB, CNB, BNB, vectorizer, text)
    #     #support_vector_machine(SVM, optimized_vectorizer, text)
    #     convolutional_neural_network(CNN, tokenizer, text)
    #     # recurrent_neural_network(RNN, tokenizer, text)


    # text = twitter.retrieve_tweets()
    # print(text)
    #
    # naive_bayes(MNB, CNB, BNB, vectorizer, text)
    # support_vector_machine(SVM, optimized_vectorizer, text)
    # convolutional_neural_network(CNN, tokenizer, text)
    # #recurrent_neural_network(RNN, tokenizer, text)
    # quit()

    text = twitter.retrieve_tweets()

    review = (naive_bayes(MNB, CNB, BNB, vectorizer, text))
    review.append(support_vector_machine(SVM, optimized_vectorizer, text))
    review.append(convolutional_neural_network(CNN, tokenizer, text))
    review.append(recurrent_neural_network(RNN, tokenizer, text))

    return review, text


#loader()