from sklearn import metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from algoritms.base import BaseModel


class Classifier(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._label = None
        self._features = None
        self._data = None
        self._target = None

        # train the model with given data set
    def train(self, data):
        ts = datetime.datetime.now()
        self._features = data["features"]
        self._label = data["label"]
        # self._model.fit(self._features, self._label)

        result = dict()
        result["metricas"] = list()

        self._data = data["data"]; ## posee el conjuto de datos completo
        self._features = data["features"] # X
        self._target = data["label"] # Y
        #self._model.fit(self._features, self._target)

        print ("muestra el data. describe")
        describe = self._data.describe() ## TODO si muestra algo mandarlo a la pantalla AttributeError: 'dict' object has no attribute 'describe
        print (describe)
        item = dict()
        item["descripcion"] = str(describe)
        item["metrica"] = "Data.describe()"
        result["metricas"].append(item)

        print ("muestra el data. corr()")
        corr = self._data.corr() ## TODO si anda mostrar el grafico
        item = dict()
        item["descripcion"] = str(corr)
        item["metrica"] = "Correlacion de los datos"
        result["metricas"].append(item)
        print (corr)




        ax = plt.figure(figsize=(12, 12))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True)
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Matriz de correlacion: {0}'.format(1)
        plt.title(all_sample_title, size=15);
        # figure.show()
        ax.savefig('MatrizCorrelacion.png')


        scores = cross_val_score(self._model, self._features, self._target, cv=5) ##cv indica en cuanto particiona el conjunto de datos
        print ("muestra el score pero del cross validation")
        item = dict()
        item["descripcion"] = str(scores.mean())
        item["metrica"] = "CrossValidation_mean"
        result["metricas"].append(item)
        item = dict()
        item["descripcion"] = str(scores)
        item["metrica"] = "CrossValidation_scores"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(self._features)
        item["metrica"] = "Datos de X completos 100%"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(self._target)
        item["metrica"] = "Datos de Y completos 100%"
        result["metricas"].append(item)

        ### REGION PRUEBA DE METRICAS###

        # Split the data into test and training (30% for test)
        X_train, X_test, Y_train, Y_test = train_test_split(self._features, self._target, test_size=0.3)

        # ENTRENAR USANDO el 70%
        self._model = self._model.fit(X_train, Y_train)
        # item = dict()
        # item["descripcion"] = str(self._model)
        # item["metrica"] = "self_model"
        # result["metricas"].append(item)


        accuracy = self._model.score(X_test, Y_test)
        item = dict()
        item["descripcion"] = str(accuracy)
        item["metrica"] = "exactitud(accuracy)"
        result["metricas"].append(item)


        print ('Accuracy(exactitud): ' + str(accuracy))

        print ('Datos de X: ' + str(self._features))
        print ('Datos de Y: ' + str(self._target))

        print ('x e y de test aplicados al modelo')

        print (X_test)

        item = dict()
        item["descripcion"] = str(X_train)
        item["metrica"] = "70% X entrenamiento"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(Y_train)
        item["metrica"] = "70% Y engtrenamiento"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(X_test)
        item["metrica"] = "30% X test"
        result["metricas"].append(item)

        item = dict()
        print (Y_test)
        item["descripcion"] = str(Y_test)
        item["metrica"] = "30% Y test"
        result["metricas"].append(item)

        prediction = self._model.predict(X_test)

        acc = metrics.accuracy_score(Y_test, prediction)

        item = dict()
        print('accuracy_score' + str(acc))
        item["descripcion"] = str(acc)
        item["metrica"] = "sklearn.metrics.accuracy_score"
        result["metricas"].append(item)

        confMatr = metrics.confusion_matrix(Y_test, prediction);
        item = dict()
        print(' metrics.confusion_matrix: ' + str(confMatr))
        item["descripcion"] = str(confMatr)
        item["metrica"] = "sklearn. metrics.confusion_matrix"
        result["metricas"].append(item)

        figure = plt.figure(figsize=(100, 100))
        sns.heatmap(confMatr, annot=True, fmt=".1f", linewidths=3, square=True, cmap='Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score: {0}'.format(acc)
        plt.title(all_sample_title, size=15);
        # figure.show()
        figure.savefig('ConfusionMatrix.png')

        item = dict()
        print ('predeciendo el X_test' + str(prediction))
        item["descripcion"] = str(prediction)
        item["metrica"] = "Y predecido con 70% datos"
        result["metricas"].append(item)

        # Measure - Since this is a regression problem, we will use the r2 score metric.
        scoreR2 = metrics.r2_score(Y_test, prediction)
        item = dict()
        item["descripcion"] = str(scoreR2)
        item["metrica"] = "score_r2_metric"
        result["metricas"].append(item)
        print (scoreR2)

        # PARA MEDIR TOMO Y_test PORQUE ES EL VALOR REAL PURO(del
        # 30% de los datos, los cuales no se usaron hasta ahora),
        # PARA PODER ESTABLECER LA RELACION CON LOS DATOS PREDECIDOS
        evs = metrics.explained_variance_score(Y_test, prediction)
        mae = metrics.mean_absolute_error(Y_test, prediction)
        mse = metrics.mean_squared_error(Y_test, prediction)
        mslg = metrics.mean_squared_log_error(Y_test, prediction)
        mene = metrics.median_absolute_error(Y_test, prediction)
        # RMSE - Root Mean Squared Error (RMSE) es la raiz cuadrada the mean of the squared errors:
        # MSE is more popular than MAE because MSE "punishes" larger errors. But, RMSE is even more popular than MSE because RMSE is interpretable in the "y" units.
        rmse = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
        print(rmse)

        item = dict()
        item["descripcion"] = str(rmse)
        item["metrica"] = "Root Mean Squared Error (RMSE)"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(evs)
        item["metrica"] = "explained_variance_score"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(mae)
        item["metrica"] = "mean_absolute_error"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(mse)
        item["metrica"] = "mean_squared_error"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(mslg)
        item["metrica"] = "mean_squared_log_error"
        result["metricas"].append(item)

        item = dict()
        item["descripcion"] = str(mene)
        item["metrica"] = "median_absolute_error"
        result["metricas"].append(item)

        tf = datetime.datetime.now()
        tf = tf - ts
        item = dict()
        item["descripcion"] = str(tf)
        item["metrica"] = "time"
        result["metricas"].append(item)


        print (evs)
        print (mae)
        print (mse)
        print (mslg)
        print (mene)

        # The coefficients
        # print('Coefficients: \n', self._model.coef_)
        # item = dict()
        # item["descripcion"] = str(self._model.coef_)
        # item["metrica"] = "coeficiente del modelo luego de la prediccion"
        # result["metricas"].append(item)

        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(Y_test, prediction))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(Y_test, prediction))
        ### END REGION PRUEBA DE METRICAS###

        return result["metricas"]

    # train the model with given data set
    def getParameterDef(self):
        pass

    def setParameter(self, parameter):
        pass

    # predict the model with given dataset
    def predict(self, data):
        return self._model.predict(data)

    def predictViz(self, scale):
        # Predict Viz only available for two dimensional dataset
        if len(self._features[0]) != 2:
            return None

        result = dict()
        result["predict"] = list()
        result["data"] = list()

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

        for i in range(0, len(self._label) - 1):
            record = dict()
            record["x"] = self._features[i][0]
            record["y"] = self._features[i][1]
            record["label"] = self._label[i]
            result["data"].append(record)

        return result
