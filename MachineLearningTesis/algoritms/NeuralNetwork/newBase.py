from algoritms.base import BaseModel


class NeuralNetwork_New(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._label = None
        self._features = None
        self._columns = None

    # train the model with given data set
    def train(self, datafm, data):
        self._features = data["features"]
        self._label = data["label"]
        self._columns = data["columns"]
        # return self._model.fit(self._features, self._label)
        return self._model.fit(datafm,self._features, self._label, self._columns) # ENVIO UN DATAFRAME A LA CAPA LSTMNEURALNETWORK
    # train the model with given data set
    def getParameterDef(self):
        pass

    def setParameter(self, parameter):
        pass

    # predict the model with given dataset
    def predict(self, data):
        return self._model.predict(data)

    def predictViz(self, scale):
        print ("father")
        result = dict()


        return result

