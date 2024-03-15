import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn import metrics
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plot
from sklearn.preprocessing import label_binarize


def support_vector_machine(dataset, text_counts, vectorizer, *args):
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    train_vectors = vectorizer.fit_transform(text_train)
    test_vectors = vectorizer.transform(text_test)

    joblib.dump(vectorizer, "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/OptimizedVectorizer.pkl")

    SVM = LinearSVC(random_state=0, tol=1e-5)
    SVM.fit(train_vectors, sentiment_train)

    predictionSVM = SVM.predict(test_vectors)

    sentiment_test_binary = label_binarize(sentiment_test, classes=SVM.classes_)
    fpr, tpr, _ = roc_curve(sentiment_test_binary.ravel(), SVM.decision_function(test_vectors).ravel())
    roc_svm = roc_auc_score(sentiment_test_binary, SVM.decision_function(test_vectors), average='macro')

    cm_svm = confusion_matrix(sentiment_test, predictionSVM, labels=SVM.classes_)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=SVM.classes_)

    disp_svm.plot()
    #plot.title("Support Vector Machine (SVM) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/SupportVectorMachine/svm_confusion_matrix.png')

    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/SupportVectorMachine/svm_roc_curve.png')

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/SupportVectorMachine/metrics.txt','w') as the_file:
        the_file.write(
            "SVM Accuracy: " + str(metrics.accuracy_score(predictionSVM, sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: " + str(
                classification_report(sentiment_test, predictionSVM, output_dict=True)["4"]) + "\n" +
            "Negative: " + str(
                classification_report(sentiment_test, predictionSVM, output_dict=True)["0"]) + "\n\n"
        )
        the_file.write(str(roc_svm))

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("SVM: " + str(metrics.accuracy_score(predictionSVM, sentiment_test)) +  "\n")

    joblib.dump(SVM,
                "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/SupportVectorMachine/SupportVectorMachine.pkl")
