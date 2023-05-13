import joblib
from sklearn import metrics
from keras.utils import pad_sequences
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, MaxPooling1D, Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from os.path import exists
import matplotlib.pyplot as plot
import numpy as np


def convolutional_neural_network(dataset, text_counts, vectorizer, *args):
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    vocab_size = 10000
    if exists("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl"):
        tokenizer = joblib.load("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")
    else:

        tokenizer = Tokenizer(vocab_size)
        tokenizer.fit_on_texts(text_train)
        joblib.dump(tokenizer, "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")

    sequence_train = tokenizer.texts_to_sequences(text_train)
    sequence_test = tokenizer.texts_to_sequences(text_test)

    data_train = pad_sequences(sequence_train)
    t = data_train.shape[1]
    data_test = pad_sequences(sequence_test, maxlen=t)

    # define model
    CNN = Sequential()
    CNN.add(Embedding(len(tokenizer.word_index) + 1, 5, input_length=t))  # T = max_length # First param = vocab size
    CNN.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
    CNN.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
    CNN.add(MaxPooling1D(pool_size=2))
    CNN.add(Flatten())
    CNN.add(Dense(10, activation='relu'))
    CNN.add(Dense(5, activation='sigmoid'))

    CNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    history = CNN.fit(data_train, sentiment_train, validation_data=(data_test, sentiment_test), epochs=5, batch_size=100)

    # (history.history['accuracy'])
    # print(history.history.keys())
    # summarize history for accuracy
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.title('Accuracy Evolution')
    plot.ylabel('accuracy')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    #plot.show()
    plot.savefig(
       'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/accuracy_evolution.png')

    # summarize history for loss
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title('Loss Evolution')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    #plot.show()
    plot.savefig(
       'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/loss_evolution.png')

    predictionCNN = CNN.predict(data_test)

    cm_cnn = confusion_matrix(sentiment_test, np.argmax(predictionCNN,axis=1))

    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_cnn)

    disp_svm.plot()
    plot.title("Convolutional Neural Networks (CNN) - Metrics")
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/cnn_confusion_matrix.png')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/confusion_matrix/CNN.png')

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/metrics.txt','w') as the_file:
        the_file.write(
            "CNN Accuracy: " + str(metrics.accuracy_score(np.argmax(predictionCNN, axis=1), sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: " + str(
                classification_report(sentiment_test, np.argmax(predictionCNN, axis=1), output_dict=True)["4"]) + "\n" +
            "Negative: " + str(
                classification_report(sentiment_test, np.argmax(predictionCNN, axis=1), output_dict=True)["0"]) + "\n\n"
        )

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("CNN: " + str(metrics.accuracy_score(np.argmax(predictionCNN, axis=1), sentiment_test)) +  "\n")

    CNN.save(
        "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h5")
