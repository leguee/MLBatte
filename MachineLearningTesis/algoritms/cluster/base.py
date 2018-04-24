from algoritms.base import BaseModel


class Cluster(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._features = None

    # train the model with given data set
    def train(self, data):
        self._features = data["train"]
        self._model.fit(self._features)

    # train the model with given data set
    def getParameterDef(self):
        pass

    def setParameter(self, parameter):
        pass

    # predict the model with given dataset
    def predict(self, data):
        return self._model.predict(data)

    def predictViz(self, scale):
        # Predict Viz only available for one dimensional dataset
        if len(self._features[0]) < 2:
            return None

        result = dict()
        result["predict"] = list()
        result["data"] = list()

        predict_train = self.predict(self._features)

        for i in range(0, len(self._features)):
            item = dict()
            item["x"] = self._features[i][0]
            item["y"] = self._features[i][1]
            item["label"] = predict_train[i]
            result["data"].append(item)

        # TODO leverage pandas to do this?
        aarange = dict()
        aarange["xmin"] = self._features[0][0]
        aarange["xmax"] = self._features[0][0]

        aarange["ymin"] = self._features[0][1]
        aarange["ymax"] = self._features[0][1]

        for item in self._features:
            if item[0] > aarange["xmax"]:
                aarange["xmax"] = item[0]
            if item[0] < aarange["xmin"]:
                aarange["xmin"] = item[0]
            if item[1] > aarange["ymax"]:
                aarange["ymax"] = item[1]
            if item[1] < aarange["ymin"]:
                aarange["ymin"] = item[1]

        xstep = (float(aarange["xmax"]) - float(aarange["xmin"])) / scale
        ystep = (float(aarange["ymax"]) - float(aarange["ymin"])) / scale

        for x in range(0, scale):
            dx = aarange["xmin"] + x * xstep
            dy = aarange["ymin"]
            for y in range(0, scale):
                dy = dy + ystep
                onePredict = self.predict([[dx, dy]])
                record = dict()
                record["x"] = dx
                record["y"] = dy
                record["label"] = onePredict[0]
                result["predict"].append(record)

        return result
