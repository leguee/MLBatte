from algoritms.base import BaseModel


class NeuralNetwork(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._label = None
        self._features = None

    # train the model with given data set
    def train(self, data):
        self._features = data["features"]
        self._label = data["label"]
        self._model.fit(self._features, self._label)

    # train the model with given data set
    def getParameterDef(self):
        pass

    def setParameter(self, parameter):
        pass

    # predict the model with given dataset
    def predict(self, data):
        return self._model.predict(data)

    def predictViz(self, scale):

        result = dict()
        return result

