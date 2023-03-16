######################################################
#                                                    #
# This version utilizes Tensorflow model for testing #
#                                                    #
######################################################


import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import losses

# Loading Model
model_new = tf.keras.models.load_model('twitterNLP_Model')
model_new.summary()


# Model activation
probability_model = keras.Sequential([
                        model_new,
                        layers.Activation('sigmoid')
                        ])


# Input Testing
examples = [
    "great, great, great",
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible...",
    "terrible, terrible, terrible"
]

while True:
    print(probability_model.predict([input("Enter phrase:")]))





