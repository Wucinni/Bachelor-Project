import joblib
from keras.layers import Conv1D, MaxPooling1D, Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plot
import numpy as np
from os.path import exists
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics


def convolutional_neural_network(dataset, text_counts, vectorizer, *args):
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    # Load or Create Tokenizer
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
    plot.close()

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
    plot.close()

    predictionCNN = CNN.predict(data_test)

    binary_sentiment_test = [1 if label == 4 else 0 for label in sentiment_test]
    fpr, tpr, _ = roc_curve(sentiment_test, predictionCNN[:, 1], pos_label=4)
    roc_auc = auc(fpr, tpr)

    cm_cnn = confusion_matrix(sentiment_test, np.argmax(predictionCNN, axis=1))
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_cnn)

    disp_svm.plot()
    # plot.title("Convolutional Neural Networks (CNN) - Metrics")
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/cnn_confusion_matrix.png')
    plot.close()

    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(
        'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/cnn_roc_curve.png')
    #plot.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

    #plot.xlim([0.0, 1.0])
    #plot.ylim([0.0, 1.05])
    #plot.xlabel('False Positive Rate')
    #plot.ylabel('True Positive Rate')
    #plot.savefig(
    #    'C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/ConvolutionalNeuralNetwork/cnn_roc_curve.png')

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
        the_file.write(str(roc_auc))

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("CNN: " + str(metrics.accuracy_score(np.argmax(predictionCNN, axis=1), sentiment_test)) +  "\n")


    CNN.save(
        "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h5")
