from sklearn.naive_bayes import GaussianNB
from algoritms.classification.base import Classifier


class NBayesClassifier(Classifier):

    def __init__(self):
        Classifier.__init__(self)
        self._name = "Bayes"
        self._model = GaussianNB()
