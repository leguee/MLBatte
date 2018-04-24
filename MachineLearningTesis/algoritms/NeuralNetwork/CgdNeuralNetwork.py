from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from neupy import environment
from neupy import algorithms,layers
from algoritms.NeuralNetwork.base import NeuralNetwork

class CgdNeuralNetwork:

    def fit(self,data,target):
        data_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.MinMaxScaler()

        data = data_scaler.fit_transform(data)
        target = target_scaler.fit_transform(target.reshape(-1,1))

        environment.reproducible()
        x_train,x_test,y_train,y_test = train_test_split(data,target,train_size=0.85)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        print (x_test)

        cgnet = algorithms.ConjugateGradient(
            connection=[
                layers.Input(2),
                layers.Sigmoid(10),
                layers.Sigmoid(1),
            ],
            search_method = 'golden',
            show_epoch=25,
            verbose=True,
            addons=[algorithms.LinearSearch],
        )
        cgnet.train(x_train,y_train,x_test,y_test,epochs=100)
        self._model = cgnet
        return self

    def predict(self,data):
        y_predict = self._model.predict(data).round(1).tolist()
        return y_predict

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None