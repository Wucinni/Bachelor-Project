#############################################
#                                           #
#   This script manages training for the    #
#  models and commandline testing through   #
#               other scripts               #
#                                           #
#############################################


import _4_trainer as trainer
import _4_tester as tester


def main():
    """
        This is the main function used in development to train and test models in terminal
        input - None
        output - None
    """

    # Uncomment next line to train the models; models, dataset location and name can be changed in _4_trainer_config.py
    # trainer.run()

    # Uncomment 5th next line to test the models by giving input from code;
    # Use the Syntax tester.run_model(model="model_name", query="query")
    # Models name are "model_1_Naive-Bayes", "model_2_SVM", "model_3_CNN", "model_4_RNN" or blank for default CNN
    # Output is a Tuple of Lists -> (Sentiment, Text, ID) - (STR, STR, STR(or INT in some cases))
    # tester.run_model()

    pass


if __name__ == "__main__":
    main()
