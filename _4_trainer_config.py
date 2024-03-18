#############################################
#                                           #
#   Configuration file for training script  #
#                                           #
#      Change model and training data       #
#                                           #
#############################################


import os


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)

data_location = path[:len(path) - len(filename)] + "\\_1_1600000tweets.csv"
dataset_name = "_1_1600000tweets.csv"  # This one is the result of preprocessing
original_dataset_name = "_1_1600000.processed.noemoticon.csv"  # This one is not cleaned


algorithms = {
    # "_2_NaiveBayes": ["naive_bayes"],
    # "_2_SupportVectorMachine": ["support_vector_machine"],
    # "_2_ConvolutionalNeuralNetwork": ["convolutional_neural_network"],
    # "_2_RecurrentNeuralNetwork": ["recurrent_neural_network"]
}
