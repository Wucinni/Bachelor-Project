#############################################
#                                           #
#   This script trains the model for the    #
#       Recurrent Neural Network            #
#                                           #
#  It also saves model locally and retains  #
#       its metrics in a text file          #
#                                           #
#############################################


import joblib
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plot
import numpy as np
import os
from os.path import exists
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)


def recurrent_neural_network(dataset, text_counts, vectorizer, *args):
    """
        Function creates a neural network and then trains it and saves performance metrics
        input - dataset; Type Pandas DataFrame
              - text_counts; Type Compressed Sparse Row Matrix -> number of apparitions for a word / token
              - vectorizer; Type Matrix of TF-IDF features
              - args; Type None -> Unused; implemented for future updates
        output - None
    """

    # Split data into training and testing
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    # Load or Create Tokenizer
    # It is used to "Learn the vocabulary"
    vocab_size = 10000
    if exists(path[:len(path)-len(filename)] + "/Models/Tokenizer.pkl"):
        tokenizer = joblib.load(path[:len(path)-len(filename)] + "/Models/Tokenizer.pkl")
    else:
        tokenizer = Tokenizer(vocab_size)
        tokenizer.fit_on_texts(text_train)
        joblib.dump(tokenizer, path[:len(path)-len(filename)] + "/Models/Tokenizer.pkl")

    # Create sequences of integers from text (tokens)
    train_sequences = tokenizer.texts_to_sequences(text_train)
    test_sequences = tokenizer.texts_to_sequences(text_test)

    # Pad the sequences -> get all of them to the same size
    data_train_padded = pad_sequences(train_sequences)
    train_data_length = data_train_padded.shape[1]
    data_test_padded = pad_sequences(test_sequences, maxlen=train_data_length)

    # define model
    RNN = Sequential()
    # Create sequence of dense vector representation based on input
    RNN.add(Embedding(len(tokenizer.word_index) + 1, 5, input_length=train_data_length))  # First param = vocab size
    # Add recurrent layers which saves revious steps in calculations
    # Relu function gives max(0, x) -> no negative values
    RNN.add(SimpleRNN(16, activation='relu', return_sequences=True))
    RNN.add(SimpleRNN(16, activation='relu'))
    # Add fully connected layers of neurons for output classification
    RNN.add(Dense(10, activation='relu'))  # Return 0 for negative otherwise value
    RNN.add(Dense(5, activation='sigmoid'))  # Classify output into 0 and 1

    # Prepare the model for training
    RNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model and save its metrics
    history = RNN.fit(data_train_padded, sentiment_train, validation_data=(data_test_padded, sentiment_test), epochs=5, batch_size=100)

    # summarize history for accuracy
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.title('Accuracy Evolution')
    plot.ylabel('accuracy')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    # plot.show()
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/RecurrentNeuralNetwork/accuracy_evolution.png')
    plot.close()

    # summarize history for loss
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title('Loss Evolution')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    # plot.show()
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/RecurrentNeuralNetwork/loss_evolution.png')
    plot.close()

    # Test model on test data
    predictionRNN = RNN.predict(data_test_padded)

    # Create performance metrics
    # Calculate ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(sentiment_test, predictionRNN[:, 1], pos_label=4)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Calculate confusion matrix
    confusion_matrix_rnn = confusion_matrix(sentiment_test, np.argmax(predictionRNN, axis=1))
    image_confusion_matrix_rnn = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rnn)

    # Plot and save Confusion Matrix
    image_confusion_matrix_rnn.plot()
    # plot.title("Recurrent Neural Networks (RNN) - Metrics")
    plot.savefig(path[:len(path)-len(filename)] + '/Models/Results/RecurrentNeuralNetwork/rnn_confusion_matrix.png')
    plot.close()

    # Plot and save ROC Curve
    RocCurveDisplay(fpr=false_positive_rate, tpr=true_positive_rate).plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(path[:len(path)-len(filename)] + '/Models/Results/RecurrentNeuralNetwork/rnn_roc_curve.png')

    # Save metrics of model to text files
    with open(path[:len(path) - len(filename)] + '/Models/Results/RecurrentNeuralNetwork/metrics.txt', 'w') as file:
        file.write(
            "RNN Accuracy: " + str(metrics.accuracy_score(np.argmax(predictionRNN, axis=1), sentiment_test)) + "\n"
        )
        file.write(
            "Positive: " + str(
                classification_report(sentiment_test, np.argmax(predictionRNN, axis=1), output_dict=True)["4"]) + "\n" +
            "Negative: " + str(
                classification_report(sentiment_test, np.argmax(predictionRNN, axis=1), output_dict=True)["0"]) + "\n\n"
        )
        file.write(str(roc_auc))

    with open(path[:len(path) - len(filename)] + '/Models/Results/accuracy.txt', 'a') as file:
        file.write("RNN: " + str(metrics.accuracy_score(np.argmax(predictionRNN, axis=1), sentiment_test)) + "\n")

    # Save model locally
    RNN.save(path[:len(path) - len(filename)] + "/Models/RecurrentNeuralNetwork/RecurrentNeuralNetwork.h5")
