import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

    predictionMultinomial_binary = multinomial.predict(text_test)
    predictionComplement_binary = complement.predict(text_test)
    predictionBernoulli_binary = bernoulli.predict(text_test)

    predictionMultinomial = multinomial.predict_log_proba(text_test)
    predictionComplement = complement.predict_log_proba(text_test)
    predictionBernoulli = bernoulli.predict_log_proba(text_test)

    predictionMultinomial = np.exp(predictionMultinomial)
    predictionComplement = np.exp(predictionComplement)
    predictionBernoulli = np.exp(predictionBernoulli)

    scaler = MinMaxScaler()

    predictionMultinomial = scaler.fit_transform(predictionMultinomial)
    predictionComplement = scaler.fit_transform(predictionComplement)
    predictionBernoulli = scaler.fit_transform(predictionBernoulli)

    save_model(multinomial, "MultinomialNaiveBayes.pkl")
    save_model(complement, "ComplementNaiveBayes.pkl")
    save_model(bernoulli, "BernoulliNaiveBayes.pkl")

    #print(metrics.accuracy_score(predictionMultinomial, sentiment_test))
    #print(metrics.accuracy_score(predictionComplement, sentiment_test))
    #print(metrics.accuracy_score(predictionBernoulli, sentiment_test))

    cm_mnb = confusion_matrix(sentiment_test, predictionMultinomial_binary, labels=multinomial.classes_)
    cm_bnb = confusion_matrix(sentiment_test, predictionBernoulli_binary, labels=multinomial.classes_)
    cm_cnb = confusion_matrix(sentiment_test, predictionComplement_binary, labels=multinomial.classes_)

    disp_mnb = ConfusionMatrixDisplay(confusion_matrix=cm_mnb, display_labels=multinomial.classes_)
    disp_bnb = ConfusionMatrixDisplay(confusion_matrix=cm_bnb, display_labels=multinomial.classes_)
    disp_cnb = ConfusionMatrixDisplay(confusion_matrix=cm_cnb, display_labels=multinomial.classes_)

    # fpr = false pozitive rate      tpr = true pozitive rate
    fpr_mnb, tpr_mnb, _ = roc_curve(sentiment_test, predictionMultinomial[:, 1], pos_label=4)
    fpr_bnb, tpr_bnb, _ = roc_curve(sentiment_test, predictionBernoulli[:, 1], pos_label=4)
    fpr_cnb, tpr_cnb, _ = roc_curve(sentiment_test, predictionComplement[:, 1], pos_label=4)

    roc_mnb = RocCurveDisplay(fpr=fpr_mnb, tpr=tpr_mnb)
    roc_bnb = RocCurveDisplay(fpr=fpr_bnb, tpr=tpr_bnb)
    roc_cnb = RocCurveDisplay(fpr=fpr_cnb, tpr=tpr_cnb)

    disp_mnb.plot()
    #plot.title("Multinomial Naive Bayes (MNB) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/mnb_confusion_matrix.png')
    roc_mnb.plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/mnb_roc_curve.png')

    disp_bnb.plot()
    #plot.title("Bernoulli Naive Bayes (BNB) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/bnb_confusion_matrix.png')
    roc_bnb.plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/bnb_roc_curve.png')

    disp_cnb.plot()
    #plot.title("Complement Naive Bayes (CNB) - Metrics")
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/cnb_confusion_matrix.png')
    roc_cnb.plot()
    plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plot.savefig('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/cnb_roc_curve.png')

    #plot.show()

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/NaiveBayes/metrics.txt', 'w') as the_file:
        # Multinomial
        the_file.write(
            "Multinomial Accuracy: " + str(metrics.accuracy_score(predictionMultinomial_binary, sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: "+str(classification_report(sentiment_test, predictionMultinomial_binary, output_dict=True)["4"])+"\n" +
            "Negative: "+str(classification_report(sentiment_test, predictionMultinomial_binary, output_dict=True)["0"])+"\n\n"
        )

        # Complement
        the_file.write(
            "Complement Accuracy: " + str(metrics.accuracy_score(predictionComplement_binary, sentiment_test)) + "\n")
        the_file.write(
            "Positive: "+str(classification_report(sentiment_test, predictionComplement_binary, output_dict=True)["4"])+"\n" +
            "Negative: "+str(classification_report(sentiment_test, predictionComplement_binary, output_dict=True)["0"])+"\n\n"
        )

        # Bernoulli
        the_file.write(
            "Bernoulli Accuracy: " + str(metrics.accuracy_score(predictionBernoulli_binary, sentiment_test)) + "\n"
        )
        the_file.write(
            "Positive: " +str(classification_report(sentiment_test, predictionBernoulli_binary, output_dict=True)["4"])+"\n" +
            "Negative: " +str(classification_report(sentiment_test, predictionBernoulli_binary, output_dict=True)["0"])+"\n\n"
        )

        the_file.write(
            str(roc_auc_score(sentiment_test, predictionMultinomial[:, 1])) + "\n" +
            str(roc_auc_score(sentiment_test, predictionComplement[:, 1])) + "\n" +
            str(roc_auc_score(sentiment_test, predictionBernoulli[:, 1]))
        )

    with open('C:/Users/Dennis/PycharmProjects/Bachelor-Project/Models/Results/accuracy.txt', 'a') as the_file:
        the_file.write("MNB: " + str(metrics.accuracy_score(predictionMultinomial_binary, sentiment_test)) +  "\n")
        the_file.write("CNB: " + str(metrics.accuracy_score(predictionComplement_binary, sentiment_test)) +  "\n")
        the_file.write("BNB: " + str(metrics.accuracy_score(predictionBernoulli_binary, sentiment_test)) +  "\n")
