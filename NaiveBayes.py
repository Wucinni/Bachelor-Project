import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plot


def save_model(model, name):
    joblib.dump(model, "C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/NaiveBayes/" + name)


def naive_bayes(dataset, text_counts, vectorizer, *args):
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(text_counts, dataset['sentiment'],
                                                                              test_size=0.25, random_state=42)
    multinomial = MultinomialNB()
    complement = ComplementNB()
    bernoulli = BernoulliNB()

    multinomial.fit(text_train, sentiment_train)
    complement.fit(text_train, sentiment_train)
    bernoulli.fit(text_train, sentiment_train)

    predictionMultinomial = multinomial.predict(text_test)
    predictionComplement= complement.predict(text_test)
    predictionBernoulli = bernoulli.predict(text_test)

    save_model(multinomial, "MultinomialNaiveBayes.pkl")
    save_model(complement, "ComplementNaiveBayes.pkl")
    save_model(bernoulli, "BernoulliNaiveBayes.pkl")

    #print(metrics.accuracy_score(predictionMultinomial, sentiment_test))
    #print(metrics.accuracy_score(predictionComplement, sentiment_test))
    #print(metrics.accuracy_score(predictionBernoulli, sentiment_test))

    cm_mnb = confusion_matrix(sentiment_test, predictionMultinomial, labels=multinomial.classes_)
    cm_bnb = confusion_matrix(sentiment_test, predictionBernoulli, labels=multinomial.classes_)
    cm_cnb = confusion_matrix(sentiment_test, predictionComplement, labels=multinomial.classes_)

    disp_mnb = ConfusionMatrixDisplay(confusion_matrix=cm_mnb, display_labels=multinomial.classes_)
    disp_bnb = ConfusionMatrixDisplay(confusion_matrix=cm_bnb, display_labels=multinomial.classes_)
    disp_cnb = ConfusionMatrixDisplay(confusion_matrix=cm_cnb, display_labels=multinomial.classes_)

    disp_mnb.plot()
    plot.title("Multinomial Naive Bayes (MNB) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/mnb_confusion_matrix.png')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/confusion_matrix/MNB.png')

    disp_bnb.plot()
    plot.title("Bernoulli Naive Bayes (BNB) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/bnb_confusion_matrix.png')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/confusion_matrix/BNB.png')

    disp_cnb.plot()
    plot.title("Complement Naive Bayes (CNB) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/cnb_confusion_matrix.png')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/confusion_matrix/CNB.png')

    #plot.show()

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/metrics.txt', 'w') as the_file:
        # Multinomial
        the_file.write(
            "Multinomial Accuracy: " + str(metrics.accuracy_score(predictionMultinomial, sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: "+str(classification_report(sentiment_test, predictionMultinomial, output_dict=True)["4"])+"\n" +
            "Negative: "+str(classification_report(sentiment_test, predictionMultinomial, output_dict=True)["0"])+"\n\n"
        )

        # Complement
        the_file.write(
            "Complement Accuracy: " + str(metrics.accuracy_score(predictionComplement, sentiment_test)) + "\n")
        the_file.write(
            "Positive: "+str(classification_report(sentiment_test, predictionComplement, output_dict=True)["4"])+"\n" +
            "Negative: "+str(classification_report(sentiment_test, predictionComplement, output_dict=True)["0"])+"\n\n"
        )

        # Bernoulli
        the_file.write(
            "Bernoulli Accuracy: " + str(metrics.accuracy_score(predictionBernoulli, sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: " +str(classification_report(sentiment_test, predictionBernoulli, output_dict=True)["4"])+"\n" +
            "Negative: " +str(classification_report(sentiment_test, predictionBernoulli, output_dict=True)["0"])+"\n\n"
        )


    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("MNB: " + str(metrics.accuracy_score(predictionMultinomial, sentiment_test)) +  "\n")
        the_file.write("CNB: " + str(metrics.accuracy_score(predictionComplement, sentiment_test)) +  "\n")
        the_file.write("BNB: " + str(metrics.accuracy_score(predictionBernoulli, sentiment_test)) +  "\n")

