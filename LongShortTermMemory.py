import joblib
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import keras
from sklearn.metrics import accuracy_score
from keras.layers import Bidirectional, LSTM, SimpleRNN
from os.path import exists


def long_short_term_memory(dataset, text_counts, vectorizer, *args):
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

    train_padded = pad_sequences(train_sequences, padding='post', maxlen=200)
    test_padded = pad_sequences(test_sequences, padding='post', maxlen=200)

    # model initialization
    model = keras.Sequential([
        # vocab_size = 10000, embedding_dim = 100, max_length = 200
        keras.layers.Embedding(vocab_size, 100, input_length=200),
        # Bidirectional(LSTM(200)),
        SimpleRNN(32),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_padded, sentiment_train, epochs=5, verbose=1, validation_split=0.1, batch_size=200)

    prediction = model.predict(test_padded)

    # Get labels based on probability 1 if p>= 0.5 else 0
    pred_labels = []
    for i in prediction:
        if i >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    print("Accuracy of prediction on test set : ", accuracy_score(sentiment_test, pred_labels))

    model.save("C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/LongShortTermMemory/LongShortTermMemory.h5")
