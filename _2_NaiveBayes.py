#############################################
#                                           #
#   This script trains the models for the   #
#    Naive Bayes from which the best one    #
#                is chosen                  #
#                                           #
#       It consists of 3 models:            #
#        Multinomial Naive-Bayes            #
#        Complement Naive-Bayes             #
#        Bernoulli Naive-Bayes              #
#                                           #
#  It also saves models locally and retains #
#       its metrics in a text file          #
#                                           #
#############################################


import joblib
import matplotlib.pyplot as plot
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)


def save_model(model, model_name):
    """
        Function saves models locally
        input - model; Type SkLearn NaiveBayes Object(instance)
              - model_name; Type STR
        output - None
    """
    joblib.dump(model, path[len(path) - len(filename)] + "/Models/NaiveBayes/" + model_name)


def naive_bayes(dataset, text_counts, vectorizer, *args):
    """
        Function creates a Naive-Bayes classifier, trains it  and saves performance metrics
        input - dataset; Type Pandas DataFrame
              - text_counts; Type Compressed Sparse Row Matrix -> number of apparitions for a word / token
              - vectorizer; Type Matrix of TF-IDF features
              - args; Type None -> Unused; implemented for future updates
        output - None
    """

    # Split data into training and testing
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(text_counts, dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    # Define models
    multinomial = MultinomialNB()
    complement = ComplementNB()
    bernoulli = BernoulliNB()

    # Train models
    multinomial.fit(text_train, sentiment_train)
    complement.fit(text_train, sentiment_train)
    bernoulli.fit(text_train, sentiment_train)

    # Test models on test data
    predictionMultinomial_binary = multinomial.predict(text_test)
    predictionComplement_binary = complement.predict(text_test)
    predictionBernoulli_binary = bernoulli.predict(text_test)

    # Calculate the logarithmic probability for each label
    predictionMultinomial = multinomial.predict_log_proba(text_test)
    predictionComplement = complement.predict_log_proba(text_test)
    predictionBernoulli = bernoulli.predict_log_proba(text_test)

    # Calculate actual probabilities by exponentiation of logarithms
    predictionMultinomial = np.exp(predictionMultinomial)
    predictionComplement = np.exp(predictionComplement)
    predictionBernoulli = np.exp(predictionBernoulli)

    # Scale probabilities to obtain binary labels -> 0 and 1 instead of float
    scaler = MinMaxScaler()
    predictionMultinomial = scaler.fit_transform(predictionMultinomial)
    predictionComplement = scaler.fit_transform(predictionComplement)
    predictionBernoulli = scaler.fit_transform(predictionBernoulli)

    # Save models locally
    save_model(multinomial, "MultinomialNaiveBayes.pkl")
    save_model(complement, "ComplementNaiveBayes.pkl")
    save_model(bernoulli, "BernoulliNaiveBayes.pkl")

    # Display performance metrics for models
    # print(metrics.accuracy_score(predictionMultinomial, sentiment_test))
    # print(metrics.accuracy_score(predictionComplement, sentiment_test))
    # print(metrics.accuracy_score(predictionBernoulli, sentiment_test))

    # Calculate confusion matrix
    confusion_matrix_multinomial = confusion_matrix(sentiment_test, predictionMultinomial_binary, labels=multinomial.classes_)
    confusion_matrix_bernoulli = confusion_matrix(sentiment_test, predictionBernoulli_binary, labels=bernoulli.classes_)
    confusion_matrix_complement = confusion_matrix(sentiment_test, predictionComplement_binary, labels=complement.classes_)

    image_confusion_matrix_multinomial = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_multinomial, display_labels=multinomial.classes_)
    image_confusion_matrix_bernoulli = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_bernoulli, display_labels=bernoulli.classes_)
    image_confusion_matrix_complement = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_complement, display_labels=complement.classes_)

    # Calculate ROC Curve
    false_positive_rate_multinomial, true_positive_rate_multinomial, _ = roc_curve(sentiment_test, predictionMultinomial[:, 1], pos_label=4)
    false_positive_rate_bernoulli, true_positive_rate_bernoulli, _ = roc_curve(sentiment_test, predictionBernoulli[:, 1], pos_label=4)
    false_positive_rate_complement, true_positive_rate_complement, _ = roc_curve(sentiment_test, predictionComplement[:, 1], pos_label=4)

    roc_multinomial = RocCurveDisplay(fpr=false_positive_rate_multinomial, tpr=true_positive_rate_multinomial)
    roc_bernoulli = RocCurveDisplay(fpr=false_positive_rate_bernoulli, tpr=true_positive_rate_bernoulli)
    roc_complement = RocCurveDisplay(fpr=false_positive_rate_complement, tpr=true_positive_rate_complement)

    # Plot and save Confusion Matrix for Multinomial Model
    image_confusion_matrix_multinomial.plot()
    #plot.title("Multinomial Naive Bayes (MNB) - Metrics")
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/mnb_confusion_matrix.png')
    roc_multinomial.plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/mnb_roc_curve.png')

    # Plot and save Confusion Matrix for Bernoulli Model
    image_confusion_matrix_bernoulli.plot()
    #plot.title("Bernoulli Naive Bayes (BNB) - Metrics")
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/bnb_confusion_matrix.png')
    roc_bernoulli.plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/bnb_roc_curve.png')

    # Plot and save Confusion Matrix for Complement Model
    image_confusion_matrix_complement.plot()
    #plot.title("Complement Naive Bayes (CNB) - Metrics")
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/cnb_confusion_matrix.png')
    roc_complement.plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/cnb_roc_curve.png')

    # Save metrics of models to text files
    with open(path[:len(path) - len(filename)] + '/Models/Results/NaiveBayes/metrics.txt', 'w') as file:
        # Multinomial
        file.write(
            "Multinomial Accuracy: " + str(metrics.accuracy_score(predictionMultinomial_binary, sentiment_test)) + "\n"
        )
        file.write(
            "Positive: "+str(classification_report(sentiment_test, predictionMultinomial_binary, output_dict=True)["4"])+"\n" +
            "Negative: "+str(classification_report(sentiment_test, predictionMultinomial_binary, output_dict=True)["0"])+"\n\n"
        )

        # Complement
        file.write(
            "Complement Accuracy: " + str(metrics.accuracy_score(predictionComplement_binary, sentiment_test)) + "\n")
        file.write(
            "Positive: "+str(classification_report(sentiment_test, predictionComplement_binary, output_dict=True)["4"])+"\n" +
            "Negative: "+str(classification_report(sentiment_test, predictionComplement_binary, output_dict=True)["0"])+"\n\n"
        )

        # Bernoulli
        file.write(
            "Bernoulli Accuracy: " + str(metrics.accuracy_score(predictionBernoulli_binary, sentiment_test)) + "\n"
        )
        file.write(
            "Positive: " +str(classification_report(sentiment_test, predictionBernoulli_binary, output_dict=True)["4"])+"\n" +
            "Negative: " +str(classification_report(sentiment_test, predictionBernoulli_binary, output_dict=True)["0"])+"\n\n"
        )

        file.write(
            str(roc_auc_score(sentiment_test, predictionMultinomial[:, 1])) + "\n" +
            str(roc_auc_score(sentiment_test, predictionComplement[:, 1])) + "\n" +
            str(roc_auc_score(sentiment_test, predictionBernoulli[:, 1]))
        )

    with open(path[:len(path) - len(filename)] + '/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("MNB: " + str(metrics.accuracy_score(predictionMultinomial_binary, sentiment_test)) + "\n")
        the_file.write("CNB: " + str(metrics.accuracy_score(predictionComplement_binary, sentiment_test)) + "\n")
        the_file.write("BNB: " + str(metrics.accuracy_score(predictionBernoulli_binary, sentiment_test)) + "\n")
