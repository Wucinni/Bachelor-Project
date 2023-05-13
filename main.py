#################################################
#                                               #
# Author: Geanaliu Andy Dennis                  #
#                                               #
#################################################

import trainer
import tester
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


def mainer():
    trainer.run()
    #tester.loader()

mainer()
