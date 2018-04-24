from algoritms.NeuralNetwork.newBase import NeuralNetwork_New
from algoritms.NeuralNetwork.LstmNeuralNetwork import LstmNeuralNetwork

class Lstm(NeuralNetwork_New):

    def __init__(self):
        NeuralNetwork_New.__init__(self)
        self._name = "Lstm"
        self._model = LstmNeuralNetwork()

    def predictViz(self, scale):
        result = dict()
        result["predict"] = list()
        result["data"] = list()
        temp = []
        temp = self.predict(self._model.x_test)
        for x in range(0,len(temp)):
            record = dict()
            record["x"] = x
            record["y"] = temp[x]
            record["label"] = "predict"
            result["predict"].append(record)

        temp = self._model.y_test.tolist()
        for x in range(0,len(temp)):
            record = dict()
            record["x"] = x
            record["y"] = temp[x]
            record["label"] = "true"
            result["data"].append(record)
        return result

