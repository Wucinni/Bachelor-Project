#############################################
#                                           #
#   Configuration file for training script  #
#                                           #
#   Change model and training data          #
#                                           #
#############################################


import os

data_location = os.getcwd() + "\\_1_1600000.processed.noemoticon.csv"
print(data_location)

algorithms = {
    # "_2_NaiveBayes": ["naive_bayes"],
    # "_2_SupportVectorMachine": ["support_vector_machine"],
    # "_2_ConvolutionalNeuralNetwork": ["convolutional_neural_network"],
    # "_2_RecurrentNeuralNetwork": ["recurrent_neural_network"]
}
