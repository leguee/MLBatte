from algoritms.NeuralNetwork.base import NeuralNetwork
from algoritms.NeuralNetwork.CgdNeuralNetwork import CgdNeuralNetwork

class Cgd(NeuralNetwork):

    def __init__(self):
        NeuralNetwork.__init__(self)
        self._name = "Cgd"
        self._model = CgdNeuralNetwork()

    def predictViz(self, scale):
        # Predict Viz only available for two dimensional dataset
        #if len(self._features[0]) != 2:
        #    return None

        result = dict()
        result["predict"] = list()
        result["data"] = list()
        temp = list()
        temp = self._model.y_test.tolist()
        for i in range(0, len(temp) - 1):
            record = dict()
            record["x"] = i
            record["y"] = temp[i]
            record["label"] = "true"
            result["data"].append(record)


        temp = self.predict(self._model.x_test)
        for i in range(0, len(temp) - 1):
            record = dict()
            record["x"] = i
            record["y"] = temp[i]
            record["label"] = "predict"
            result["predict"].append(record)
        return result


