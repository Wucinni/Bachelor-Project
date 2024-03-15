import joblib
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import keras
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, auc, RocCurveDisplay
from keras.layers import Bidirectional, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Dense, Embedding, Flatten
from os.path import exists
from keras.models import Sequential
import matplotlib.pyplot as plot
import numpy as np
from sklearn import metrics


def recurrent_neural_network(dataset, text_counts, vectorizer, *args):
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    vocab_size = 10000
    if exists("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl"):
        tokenizer = joblib.load("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")
    else:

        tokenizer = Tokenizer(vocab_size)
        tokenizer.fit_on_texts(text_train)
        joblib.dump(tokenizer, "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Tokenizer.pkl")

    train_sequences = tokenizer.texts_to_sequences(text_train)
    test_sequences = tokenizer.texts_to_sequences(text_test)

    #train_padded = pad_sequences(train_sequences, padding='post', maxlen=200)
    #test_padded = pad_sequences(test_sequences, padding='post', maxlen=200)

    train_padded = pad_sequences(train_sequences)
    t = train_padded.shape[1]
    test_padded = pad_sequences(test_sequences, maxlen=t)

    # define model
    RNN = Sequential()
    # vocab_size = size of tokenizer embedding_dim = 5, max_length = train_padded.shape[1]
    RNN.add(Embedding(len(tokenizer.word_index) + 1, 5, input_length=t))
    RNN.add(SimpleRNN(16, activation='relu', return_sequences=True))
    RNN.add(SimpleRNN(16, activation='relu'))
    RNN.add(Dense(10, activation='relu'))
    RNN.add(Dense(5, activation='sigmoid'))

    # compile model
    RNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = RNN.fit(train_padded, sentiment_train, validation_data=(test_padded, sentiment_test), epochs=5, batch_size=100)

    # summarize history for accuracy
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.title('Accuracy Evolution')
    plot.ylabel('accuracy')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    # plot.show()
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/RecurrentNeuralNetwork/accuracy_evolution.png')
    plot.close()

    # summarize history for loss
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title('Loss Evolution')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    # plot.show()
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/RecurrentNeuralNetwork/loss_evolution.png')
    plot.close()

    predictionRNN = RNN.predict(test_padded)

    fpr, tpr, _ = roc_curve(sentiment_test, predictionRNN[:, 1], pos_label=4)
    roc_auc = auc(fpr, tpr)

    cm_rnn = confusion_matrix(sentiment_test, np.argmax(predictionRNN, axis=1))
    disp_rnn = ConfusionMatrixDisplay(confusion_matrix=cm_rnn)

    disp_rnn.plot()
    # plot.title("Convolutional Neural Networks (CNN) - Metrics")
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/RecurrentNeuralNetwork/rnn_confusion_matrix.png')
    plot.close()

    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/RecurrentNeuralNetwork/rnn_roc_curve.png')

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/RecurrentNeuralNetwork/metrics.txt','w') as the_file:
        the_file.write(
            "RNN Accuracy: " + str(metrics.accuracy_score(np.argmax(predictionRNN, axis=1), sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: " + str(
                classification_report(sentiment_test, np.argmax(predictionRNN, axis=1), output_dict=True)["4"]) + "\n" +
            "Negative: " + str(
                classification_report(sentiment_test, np.argmax(predictionRNN, axis=1), output_dict=True)["0"]) + "\n\n"
        )
        the_file.write(str(roc_auc))

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("RNN: " + str(metrics.accuracy_score(np.argmax(predictionRNN, axis=1), sentiment_test)) +  "\n")

    RNN.save("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/RecurrentNeuralNetwork/RecurrentNeuralNetwork.h5")
