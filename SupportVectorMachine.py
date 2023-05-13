import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plot


def support_vector_machine(dataset, text_counts, vectorizer, *args):
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(dataset['text'], dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)

    train_vectors = vectorizer.fit_transform(text_train)
    test_vectors = vectorizer.transform(text_test)

    joblib.dump(vectorizer, "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/OptimizedVectorizer.pkl")

    SVM = LinearSVC(random_state=0, tol=1e-5)
    SVM.fit(train_vectors, sentiment_train)

    predictionSVM = SVM.predict(test_vectors)

    cm_svm = confusion_matrix(sentiment_test, predictionSVM, labels=SVM.classes_)

    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=SVM.classes_)

    disp_svm.plot()
    plot.title("Support Vector Machine (SVM) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/SupportVectorMachine/svm_confusion_matrix.png')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/confusion_matrix/SVM.png')

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

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("SVM: " + str(metrics.accuracy_score(predictionSVM, sentiment_test)) +  "\n")

    joblib.dump(SVM,
                "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/SupportVectorMachine/SupportVectorMachine.pkl")
