import datetime
import warnings
import numpy as np
from collections import deque
import time
from math import sqrt
from numpy import newaxis
from numpy import concatenate
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from pandas import DataFrame, concat
from scipy._lib.six import xrange

import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class LstmNeuralNetwork():

    warnings.filterwarnings("ignore")

    def normalise_windows(self,window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def build_model(self, layers, metricas):
        model_train = Sequential()
        model_train.add(LSTM( #  Metrica de configuracion de cada capa
            input_dim=layers[0],
            output_dim=layers[1],
            return_sequences=True))
        item = dict()
        item["descripcion"] = "imput_dim:" + str(layers[0]) + "| output_dim:" + str(layers[1]) + "| return_sequences:True"
        item["metrica"] = "add LSTM Modelo config"
        metricas["metricas"].append(item)

        model_train.add(Dropout(0.2)) #  Metrica de config dropout
        item = dict()
        item["descripcion"] = "(0.2)"
        item["metrica"] = "add Dropout Modelo config"
        metricas["metricas"].append(item)

        model_train.add(LSTM( #  Metrica de configuracion de cada capa
            layers[2],
            return_sequences=False))
        item = dict()
        item["descripcion"] = "layers-2" + str(layers[2]) + "| return_sequences:False"
        item["metrica"] = "add LSTM Modelo config"
        metricas["metricas"].append(item)

        model_train.add(Dropout(0.2)) #  Metrica dropout
        item = dict()
        item["descripcion"] = "(0.2)"
        item["metrica"] = "add Dropout Modelo config"
        metricas["metricas"].append(item)

        model_train.add(Dense( #  Metrica de configuracion de cada capa
            output_dim=layers[3]))
        item = dict()
        item["descripcion"] = "output_dim: " + str(layers[3])
        item["metrica"] = "add Dense Modelo config"
        metricas["metricas"].append(item)

        model_train.add(Activation("linear")) #  Metrica de tipo de activacion
        item = dict()
        item["descripcion"] = "linear"
        item["metrica"] = "add Activation Modelo config"
        metricas["metricas"].append(item)

        model_train.compile(loss="mse", optimizer="rmsprop") #  Metrica de configuracion de compilacion
        item = dict()
        item["descripcion"] = "loss=mse|optimizer=rmsprop"
        item["metrica"] = "compile Modelo config"
        metricas["metricas"].append(item)

        return model_train

    def predict_point_by_point(self,model, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence_full(self,model, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        curr_frame = data[0]
        predicted = []
        for i in xrange(len(data)):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        return predicted

    def predict_sequences_multiple(self,model, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in xrange(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in xrange(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def plot_results_multiple(self ,predicted_data, true_data, prediction_len):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        plt.show()

    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def fit(self,dataframe, features,data,columnas):
        # data contiene los datos de la bateria narray
        # features contiene las caracteristicas, ejem min,wifi,bl,dt tipo narray
        # dataframe contiene el conjunto de datos completo

        # PARTE NUEVA *****************************************************************************************************************************************
        # #
        # data = data.astype('float32')
        #
        # # normalize the dataset
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # data = scaler.fit_transform(data.reshape(-1, 1))
        # """
        # A simple method that we can use is to split the ordered dataset into train and test datasets. The code below
        # calculates the index of the split point and separates the data into the training datasets with 67% of the
        # observations that we can use to train our model, leaving the remaining 33% for testing the model.
        # """
        # # split into train and test sets
        # train_size = int(len(data) * 0.67)
        # test_size = len(data) - train_size
        # train, test = data[0:train_size, :], data[train_size:len(data), :]
        # print
        # "train_data_size: " + str(len(train)), " test_data_size: " + str(len(test))
        #
        # # convert an array of values into a dataset matrix
        # def create_dataset(dataset, look_back=1):
        #     dataX, dataY = [], []
        #     for i in range(len(dataset) - look_back - 1):
        #         a = dataset[i:(i + look_back), 0]
        #         dataX.append(a)
        #         dataY.append(dataset[i + look_back, 0])
        #     return np.array(dataX), np.array(dataY)
        #
        # # reshape into X=t and Y=t+1
        # look_back = 1
        # trainX, trainY = create_dataset(train, look_back)
        # testX, testY = create_dataset(test, look_back)
        #
        # # reshape input to be [samples, time steps, features]
        # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        #
        # """ The network has a visible layer with 1 input, a hidden layer with
        # 4 LSTM blocks or neurons and an output layer that makes a single value
        # prediction. The default sigmoid activation function is used for the
        # LSTM blocks. The network is trained for 100 epochs and a batch size of
        # 1 is used."""
        #
        # # create and fit the LSTM network
        # model = Sequential()
        # model.add(LSTM(4, input_dim=look_back))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
        #
        # # make predictions
        # trainPredict = model.predict(trainX)
        # testPredict = model.predict(testX)
        # # invert predictions
        # trainPredict = scaler.inverse_transform(trainPredict)
        # trainY = scaler.inverse_transform([trainY])
        # testPredict = scaler.inverse_transform(testPredict)
        # testY = scaler.inverse_transform([testY])
        # # calculate root mean squared error
        # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        # print(trainY[0])
        # print(trainPredict[:, 0])
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        # print(testY[0])
        # print(testPredict[:, 0])
        # print('Test Score: %.2f RMSE' % (testScore))
        # Fin PARTE NUEVA *****************************************************************************************************************************************

        # Inicio Parte VIEJA***************************************************************************************************************************************
        # metricas = dict()
        # metricas["metricas"] = list()
        #
        # epochs  = 1 #originalmente estaba en 1  Metrica
        # item = dict()
        # item["descripcion"] = epochs
        # item["metrica"] = "epochs"
        # metricas["metricas"].append(item)
        #
        # seq_len = 50 #  Metrica
        # item = dict()
        # item["descripcion"] = seq_len
        # item["metrica"] = "seq_len"
        # metricas["metricas"].append(item)
        #
        # sequence_length = seq_len + 1 #  Metrica
        # item = dict()
        # item["descripcion"] = sequence_length
        # item["metrica"] = "sequence_length"
        # metricas["metricas"].append(item)
        #
        # result = []
        # for index in range(len(data) - sequence_length): #
        #     result.append(data[index: index + sequence_length])
        #
        # result = self.normalise_windows(result) #
        #
        # result = np.array(result)
        #
        # row = round(0.9 * result.shape[0]) #  Metrica
        # item = dict()
        # item["descripcion"] = row
        # item["metrica"] = "'row' indica el numero de separacion entre 90%train y 10%test"
        # metricas["metricas"].append(item)
        #
        # train = result[:int(row), :]
        # np.random.shuffle(train)
        # x_train = train[:, :-1] #  Metrica
        # item = dict()
        # item["descripcion"] = str(x_train)
        # item["metrica"] = "x_train"
        # metricas["metricas"].append(item)
        #
        # y_train = train[:, -1] #  Metrica
        # item = dict()
        # item["descripcion"] = str(y_train)
        # item["metrica"] = "y_train"
        # metricas["metricas"].append(item)
        #
        # x_test = result[int(row):, :-1] #  Metrica
        # item = dict()
        # item["descripcion"] = str(x_test)
        # item["metrica"] = "x_test"
        # metricas["metricas"].append(item)
        #
        # y_test = result[int(row):, -1] #  Metrica
        # item = dict()
        # item["descripcion"] = str(y_test)
        # item["metrica"] = "y_test"
        # metricas["metricas"].append(item)
        # # for each sample.We can transform the prepared train and test input data into the expected structure using numpy.reshape() as follows:
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #  Metrica
        # item = dict()
        # item["descripcion"] = x_train
        # item["metrica"] = "x_train np.reshape"
        # metricas["metricas"].append(item)
        #
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #  Metrica
        # item = dict()
        # item["descripcion"] = x_test
        # item["metrica"] = "x_test np.reshape"
        # metricas["metricas"].append(item)
        #
        # #X_train, y_train, X_test, y_test = self.load_data(data,seq_len, True)
        # model_train = self.build_model([1, 50, 100, 1] , metricas) #  Metrica agregar estos parametros
        # item = dict()
        # item["descripcion"] = model_train
        # item["metrica"] = "model_train"
        # metricas["metricas"].append(item)
        # history = model_train.fit( #  Metrica de configuracion de fit, que parametros se le pasa
        #     x_train,
        #     y_train,
        #     batch_size=512,
        #     epochs= epochs,
        #     # nb_epoch=1,
        #     validation_split=0.05,
        #     verbose=0,
        #     shuffle=False)
        #
        # # list all data in history
        # print(history.history.keys())
        # predict = self.predict_sequences_multiple(model_train, x_test, seq_len, 50)
        #
        # self.plot_results_multiple(predict, y_test, 50)
        #
        #
        # item = dict()
        # item["descripcion"] = "512|0.05|0|false"
        # item["metrica"] = "batch_size|validation_split|verbose|shuffle"
        # metricas["metricas"].append(item)
        #
        # scores = model_train.evaluate(x_test, y_test, verbose=0)
        # print("%s: %.6f%%" % (model_train.metrics_names[0], scores * 100))
        #
        # print("%.6f%% (+/- %.6f%%)" % (np.mean(scores), np.std(scores)))
        #
        # self._model = model_train
        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_test = x_test
        # self.y_test = y_test
        #
        # # make predictions
        # trainPredict = self._model.predict(x_train)
        # testPredict = self._model.predict(x_test)
        # print(self._model)
        #
        #
        # loss, accuracy = self._model.evaluate(x_train, y_train)
        # print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100)) #  Metrica loss y accuracy
        # item = dict()
        # item["descripcion"] = "\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100)
        # item["metrica"] = "loss_accuracy"
        # metricas["metricas"].append(item)
        #
        # # 5. make predictions
        # probabilities = self._model.predict(x_train) #  Metrica
        # print (probabilities)
        # item = dict()
        # item["descripcion"] = str(probabilities)
        # item["metrica"] = "probabilities"
        # metricas["metricas"].append(item)
        #
        # # predictions = [float(round(x)) for x in probabilities] #  Metrica
        # predictions = np.round(probabilities)
        # item = dict()
        # item["descripcion"] = str(predictions)
        # item["metrica"] = "predictions"
        # metricas["metricas"].append(item)
        #
        # accuracy = self._model.mean(predictions == y_train) #  Metrica
        # print("Prediction Accuracy: %.2f%%" % (accuracy * 100))
        # item = dict()
        # item["descripcion"] = accuracy
        # item["metrica"] = "Prediction Accuracy:"
        # metricas["metricas"].append(item)
        # self.metricas = metricas["metricas"]
        #
        # # FIN REGION VIEJA

        # INICIO REGION V2*******************************************************************************************************************
        # load dataset
        ts = datetime.datetime.now()
        metricas = dict()
        metricas["metricas"] = list()

        values = np.concatenate((data.reshape(-1, 1), features), axis=1) # se unen las caracteristicas con la bateria
        # specify columns to plot
        i = 1
        # plot each column
        a = deque(columnas)
        a.rotate(1)
        plt.figure()
        for group in range(0, values.shape[1]):
            plt.subplot( values.shape[1], 1, i)
            plt.plot(values[:, group])
            plt.title(a[group], y=0.5, loc='right') # Ver si se puede agregar el nombre del atributo, pero habria que pasarselo como parametro
            i += 1

        plt.savefig('{0}_imagen_columnas.png'.format(a[-1])) #guarda la imagen
        plt.clf()  # Quita la figura actual para que no interfiera con la siguiente

        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning  ver si el tercer
        # parametro es de que columna debe agarra el campo Battery Ademas ver como hacer para que esta funcion una los
        #  parametros que eligio con los que que hay qye predecir, ya que desde el main le estoy enviando por separado en nparray,
        #  tiene que fomar lo mismo que el ejemplo y en la ultima columna poner el valor de la bateria. t y t-1
        reframed = self.series_to_supervised(scaled, 1, 1)

        cols= list()
        cols = [((j + values.shape[1]+1)) for j in range(values.shape[1]-1)]

        reframed.drop(reframed.columns[cols], axis=1, inplace=True) # elimina las columnas que no van a ser predecidas se queda con var1(t)
        print(reframed.head())

        # split into train and test sets
        values = reframed.values
        n_train_hours = int((70*values.shape[0])/100)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # design network
        model = Sequential()
        neuronas = 50  # TODO
        model.add(LSTM(neuronas, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        loss = 'mae' #TODO
        optimizacion = 'adam' # TODO
        model.compile(loss=loss, optimizer=optimizacion , metrics=['accuracy'])

        item = dict()
        item["descripcion"] = "LSTM(" +neuronas+ ", input_shape:(" + str(train_X.shape[1]) + "," + str(train_X.shape[2]) +')'
        item["metrica"] = "add LSTM Modelo config"
        metricas["metricas"].append(item)

        item = dict()
        item["descripcion"] = "Dense(1)"
        item["metrica"] = "add Dense Modelo config"
        metricas["metricas"].append(item)

        item = dict()
        item["descripcion"] = "loss="+ loss +"|optimizer="+ optimizacion
        item["metrica"] = "compile Modelo config"
        metricas["metricas"].append(item)

        # fit network
        epochs = 50 #TODO
        batch = 72 #TODO
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, validation_data=(test_X, test_y), verbose=2,
                            shuffle=False)
        item = dict()
        item["descripcion"] = epochs+"|"+batch
        item["metrica"] = "epochs|batch_size"
        metricas["metricas"].append(item)
        # plot history
        plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig('{0}_history_train_test.png'.format(a[-1])) #guarda la imagen
        plt.clf()  # Quita la figura actual para que no interfiera con la siguiente
        item = dict()
        item["descripcion"] = str(history.params)
        item["metrica"] = "history_Parametros generales"
        metricas["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(history.history['loss'])
        item["metrica"] = "(x epochs) label=Train  history_loss"
        metricas["metricas"].append(item)
        # item = dict()
        # item["descripcion"] = str(history.history['val_loss'])
        # item["metrica"] = "(x epochs) label=Test  history_val_loss"
        # metricas["metricas"].append(item)
        # make a prediction
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        plt.plot(inv_yhat)
        plt.plot(inv_y)
        plt.legend()
        plt.savefig('{0}_yPredict_yTest.png'.format(a[-1]))  # guarda la imagen
        plt.clf()  # Quita la figura actual para que no interfiera con la siguiente
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        item = dict()
        item["descripcion"] = 'Test RMSE: %.3f' % rmse
        item["metrica"] = "Test RMSE:"
        metricas["metricas"].append(item)
        # FIN REGION V2

        tf = datetime.datetime.now()
        tf = tf - ts
        item = dict()
        item["descripcion"] = str(tf)
        item["metrica"] = "time"
        metricas["metricas"].append(item)


        # # return self
        return metricas["metricas"]

    def predict(self,data):
        #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
        #predictions = lstm.predict_sequence_full(model, X_test, seq_len)
        data = np.array(data)
        predictions = self.predict_point_by_point(self._model, data)
        return predictions.tolist()

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.metricas = None




