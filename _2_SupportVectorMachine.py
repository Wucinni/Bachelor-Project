#############################################
#                                           #
#   This script trains the model for the    #
#        Support Vector Machine             #
#                                           #
#  It also saves model locally and retains  #
#       its metrics in a text file          #
#                                           #
#############################################


import joblib
import matplotlib.pyplot as plot
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize


filename = os.path.basename(__file__)
path = os.path.abspath(__file__)


def support_vector_machine(dataset, text_counts, vectorizer, *args):
    """
        Function creates a Support Vector Machine classifier, trains it and saves performance metrics
        input - dataset; Type Pandas DataFrame
              - text_counts; Type Compressed Sparse Row Matrix -> number of apparitions for a word / token
              - vectorizer; Type Matrix of TF-IDF features
              - args; Type None -> Unused; implemented for future updates
        output - None
    """

    # Split data into training and testing
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    # Make vectorizer learn the vocabulary in the training data
    train_vectors = vectorizer.fit_transform(text_train)
    # Apply same transformations without relearning the vocabulary
    test_vectors = vectorizer.transform(text_test)

    # Save the new vectorizer locally
    joblib.dump(vectorizer, path[:len(path) - len(filename)] + "/Models/OptimizedVectorizer.pkl")

    # Define model
    SVM = LinearSVC(random_state=0, tol=1e-5)

    # Train model
    SVM.fit(train_vectors, sentiment_train)

    # Test model on test data
    predictionSVM = SVM.predict(test_vectors)

    # Transform dataset values into bianry (old is 0 and 4)
    sentiment_test_binary = label_binarize(sentiment_test, classes=SVM.classes_)

    # Calculate ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(sentiment_test_binary.ravel(), SVM.decision_function(test_vectors).ravel())
    roc_svm = roc_auc_score(sentiment_test_binary, SVM.decision_function(test_vectors), average='macro')

    # Calculate confusion matrix
    confusion_matrix_svm = confusion_matrix(sentiment_test, predictionSVM, labels=SVM.classes_)
    image_confusion_matrix_svm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=SVM.classes_)

    # Plot and save Confusion Matrix
    image_confusion_matrix_svm.plot()
    #plot.title("Support Vector Machine (SVM) - Metrics")
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/SupportVectorMachine/svm_confusion_matrix.png')

    # Plot and save ROC Curve
    RocCurveDisplay(fpr=false_positive_rate, tpr=true_positive_rate).plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig(path[:len(path) - len(filename)] + '/Models/Results/SupportVectorMachine/svm_roc_curve.png')

    # Save metrics of models to text files
    with open(path[:len(path) - len(filename)] + '/Models/Results/SupportVectorMachine/metrics.txt','w') as file:
        file.write(
            "SVM Accuracy: " + str(metrics.accuracy_score(predictionSVM, sentiment_test)) + "\n"
        )
        file.write(
            "Positive: " + str(
                classification_report(sentiment_test, predictionSVM, output_dict=True)["4"]) + "\n" +
            "Negative: " + str(
                classification_report(sentiment_test, predictionSVM, output_dict=True)["0"]) + "\n\n"
        )
        file.write(str(roc_svm))

    with open(path[:len(path) - len(filename)] + '/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("SVM: " + str(metrics.accuracy_score(predictionSVM, sentiment_test)) + "\n")

    # Save model locally
    joblib.dump(SVM, path[:len(path) - len(filename)] + "/Models/SupportVectorMachine/SupportVectorMachine.pkl")
