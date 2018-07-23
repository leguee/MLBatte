from datetime import datetime
from flask import Flask, request, make_response, redirect, url_for, jsonify

import csv
import os
import os.path
import glob
import json
import traceback

from algoritms import createModel, getModel, getModelType
import datautil

UPLOAD_FOLDER = './data/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/debug')
def index_debug():
    return app.send_static_file('index_debug.html')


@app.route('/csvdata', methods=['GET', 'POST'])
def csvdata():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            # TODO, Validar que pasa si hay archivos repetidos
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return json.dumps({})
        return json.dumps({os.error:
                           'No esta permitido subir ese tipo de archivo.'})
    else:
        flist = [f.replace('./data'+os.sep, '').replace('.csv', '')
                 for f in glob.glob('./data/*.csv')]
        return json.dumps(flist)


@app.route('/data/<dataname>')
def getdata(dataname):
    fpath = './data/' + dataname + '.csv'

    headerOnly = False
    try:
        headerOnly = request.args['headerOnly']
        value = headerOnly.strip().upper()
        if value not in ("0", "FALSE", "F", "N", "NO", "NONE", ""):
            headerOnly = True
    except:
        pass

    if not os.path.isfile(fpath):
        return make_response('<h1>Archivo %s     no existe!</h1>' % fpath)
    else:
        with open(fpath, 'rb') as csvfile:
            if headerOnly:
                return jsonify(name=dataname, csv=csvfile.readline().decode("UTF-8"))
            else:
                return jsonify(name=dataname, csv=csvfile.read().decode("UTF-8"))

## CLASIFICACION
@app.route('/ml/cls/<action>', methods=['GET'])
def mlclsop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("Classification", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            ahora = datetime.now().strftime('%d%m%Y-%H%M%S')  # Obtiene fecha y hora actual
            print("Fecha y Hora:", ahora)  # Muestra fecha y hora
            f = open(ahora + '.txt', 'w')
            modelId = request.args['id']
            dataName = request.args['data']
            label = request.args['label']
            features = request.args['features'].split(",")

            model = getModel(modelId)

            # Guardo la cabecera del archivo, con datos genericos de lo que se eligio en la pantalla
            f.write('%s : %s \n' % ('Modelo de entrenamiento',str(model)))
            f.write('%s :%s \n' % ('Dataset seleccionado',str(dataName)))
            f.write('%s :%s \n' % ('Dato a predecir Y',str(label)))
            f.write('%s :%s \n' % ('Caracteristicas X',str(features)))

            datadf = datautil.load(dataName)
            labelData = datautil.getColValues(datadf, label)
            featureData = datautil.getColsValues(datadf, features)

            data = dict()
            data["features"] = featureData
            data["label"] = labelData
            data["data"] = datadf
            ##model.train(data)
            ## este es un llamado nuevo que retornaria los resultados de las diferentes metricas que se aplicaron al conjunto de datos}}
            modelPredic = model.train(data)
            f.write('\n')
            for elemento in modelPredic:
                f.write('%s \n' % elemento)
            f.close()
            return jsonify(result="Success", model=modelId, metric=str(modelPredic))
            ##return jsonify(result="Success", model=modelId) ## asi estaba el llamado anterior

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']

            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="No se soporta esta accion {}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


## REGRESION
@app.route('/ml/regression/<action>', methods=['GET'])
def mlregressionop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("Regression", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            ahora = datetime.now().strftime('%d%m%Y-%H%M%S')  # Obtiene fecha y hora actual
            print("Fecha y Hora:", ahora)  # Muestra fecha y hora
            f = open(ahora +'.txt', 'w')

            modelId = request.args['id']
            dataName = request.args['data']
            label = request.args['target']
            features = request.args['train'].split(",")

            model = getModel(modelId)

            # Guardo la cabecera del archivo, con datos genericos de lo que se eligio en la pantalla
            f.write('%s : %s \n' % ('Modelo de entrenamiento',str(model)))
            f.write('%s :%s \n' % ('Dataset seleccionado',str(dataName)))
            f.write('%s :%s \n' % ('Dato a predecir Y',str(label)))
            f.write('%s :%s \n' % ('Caracteristicas X',str(features)))

            datadf = datautil.load(dataName)
            labelData = datautil.getColValues(datadf, label)
            featureData = datautil.getColsValues(datadf, features)

            data = dict()
            data["train"] = featureData
            data["target"] = labelData
            data["data"] = datadf
            ##model.train(data)
            ## en metric mandar los datos de las predicciones de los algoritmos, hacer el que model.train devuelva los datos concatenados o en un objeto

            modelPredic = model.train(data)
            f.write('\n')
            for elemento in modelPredic:
                f.write('%s \n' % elemento)
            f.close()

            return jsonify(result="Success", model=modelId, metric=str(modelPredic))

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']

            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="No se soporta esta accion{}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


##NNeuralNetwork
@app.route('/ml/neuralnetwork/<action>', methods=['GET'])
def mlneuralnetworkop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("NeuralNetwork", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            ahora = datetime.now().strftime('%d%m%Y-%H%M%S') # Obtiene fecha y hora actual
            print("Fecha y Hora:", ahora)  # Muestra fecha y hora
            f = open(ahora + '.txt', 'w')

            modelId = request.args['id']
            dataName = request.args['data']
            label = request.args['label']
            features = request.args['features'].split(",")

            model = getModel(modelId)

            # Guardo la cabecera del archivo, con datos genericos de lo que se eligio en la pantalla
            f.write('%s : %s \n' % ('Modelo de entrenamiento', str(model)))
            f.write('%s :%s \n' % ('Dataset seleccionado', str(dataName)))
            f.write('%s :%s \n' % ('Dato a predecir Y', str(label)))
            f.write('%s :%s \n' % ('Caracteristicas X', str(features)))

            datadf = datautil.load(dataName)
            labelData = datautil.getColValues(datadf, label)
            featureData = datautil.getColsValues(datadf, features)
            features.append(ahora)
            features.append(label)
            data = dict()
            data["features"] = featureData
            data["label"] = labelData
            data["columns"] = features

            # model.train(data)

            modelPredic = model.train(datadf, data)
            f.write('\n')
            for elemento in modelPredic:
                f.write('%s \n' % elemento)
            f.close()

            return jsonify(result="Success", model=modelId, metric=str(modelPredic))

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']
            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="No se soporta esta accion {}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


##Cluster
@app.route('/ml/cluster/<action>', methods=['GET'])
def mlclusterop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("Cluster", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            modelId = request.args['id']
            dataName = request.args['data']
            features = request.args['train'].split(",")

            model = getModel(modelId)

            datadf = datautil.load(dataName)
            featureData = datautil.getColsValues(datadf, features)

            data = dict()
            data["train"] = featureData
            model.train(data)

            return jsonify(result="Success", model=modelId)

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']

            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="No se soporta esta accion {}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


@app.route('/mlmodel/list/<type>', methods=['GET'])
def mlmodel(type):
    return json.dumps(getModelType(type))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.debug = True
    app.run()
