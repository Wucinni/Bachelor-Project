#############################################
#                                           #
#   This script trains the model for the    #
#       Convolutional Neural Network        #
#                                           #
#  It also saves model locally and retains  #
#       its metrics in a text file          #
#                                           #
#############################################


import joblib
from keras.layers import Conv1D, MaxPooling1D, Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plot
import numpy as np
import os
from os.path import exists
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)


def convolutional_neural_network(dataset, text_counts, vectorizer, *args):
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
    sequence_train = tokenizer.texts_to_sequences(text_train)
    sequence_test = tokenizer.texts_to_sequences(text_test)

    # Pad the sequences -> get all of them to the same size
    data_train = pad_sequences(sequence_train)
    train_data_length = data_train.shape[1]
    data_test = pad_sequences(sequence_test, maxlen=train_data_length)

    # Define model
    CNN = Sequential()
    # Create sequence of dense vector representation based on input
    CNN.add(Embedding(len(tokenizer.word_index) + 1, 5, input_length=train_data_length))  # First param = vocab size
    # Add convolutional layers with 1 dimension
    CNN.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
    CNN.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
    # Add pooling layer to simplify data
    CNN.add(MaxPooling1D(pool_size=2))
    # Add flatten layer to convert multidimensional arrays to 1 dimension suitable for next layers
    CNN.add(Flatten())
    # Add fully connected layers of neurons for output classification
    CNN.add(Dense(10, activation='relu'))  # Return 0 for negative otherwise value
    CNN.add(Dense(5, activation='sigmoid'))  # Classify output into 0 and 1

    # Prepare the model for training
    CNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model and save its metrics
    history = CNN.fit(data_train, sentiment_train, validation_data=(data_test, sentiment_test), epochs=5, batch_size=100)

    # Summarize history for accuracy
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.title('Accuracy Evolution')
    plot.ylabel('accuracy')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    # plot.show()
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/ConvolutionalNeuralNetwork/accuracy_evolution.png')
    plot.close()

    # summarize history for loss
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title('Loss Evolution')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    # plot.show()
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/ConvolutionalNeuralNetwork/loss_evolution.png')
    plot.close()

    # Test model on test data
    predictionCNN = CNN.predict(data_test)

    # Create performance metrics
    # Calculate ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(sentiment_test, predictionCNN[:, 1], pos_label=4)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Calculate confusion matrix
    confusion_matrix_cnn = confusion_matrix(sentiment_test, np.argmax(predictionCNN, axis=1))
    image_confusion_matrix_cnn = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cnn)

    # Plot and save Confusion Matrix
    image_confusion_matrix_cnn.plot()
    # plot.title("Convolutional Neural Networks (CNN) - Metrics")
    plot.savefig(path[:len(path)-len(filename)] + '/Models/Results/ConvolutionalNeuralNetwork/cnn_confusion_matrix.png')
    plot.close()

    # Plot and save ROC Curve
    RocCurveDisplay(fpr=false_positive_rate, tpr=true_positive_rate).plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(path[:len(path)-len(filename)] + '/Models/Results/ConvolutionalNeuralNetwork/cnn_roc_curve.png')

    # Save metrics of model to text files
    with open(path[:len(path) - len(filename)] + '/Models/Results/ConvolutionalNeuralNetwork/metrics.txt', 'w') as file:
        file.write(
            "CNN Accuracy: " + str(metrics.accuracy_score(np.argmax(predictionCNN, axis=1), sentiment_test)) + "\n"
        )
        file.write(
            "Positive: " + str(
                classification_report(sentiment_test, np.argmax(predictionCNN, axis=1), output_dict=True)["4"]) + "\n" +
            "Negative: " + str(
                classification_report(sentiment_test, np.argmax(predictionCNN, axis=1), output_dict=True)["0"]) + "\n\n"
        )
        file.write(str(roc_auc))

    with open(path[:len(path) - len(filename)] + '/Models/Results/accuracy.txt', 'a') as file:
        file.write("CNN: " + str(metrics.accuracy_score(np.argmax(predictionCNN, axis=1), sentiment_test)) + "\n")

    # Save model locally
    CNN.save(path[:len(path) - len(filename)] + "/Models/ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h5")
