from algoritms.NeuralNetwork import  cgd
from algoritms.NeuralNetwork import lstm
__CATEGORY__ = ["Cgd","Lstm"]

def getNeuralNetworkByName(name):

    if name == "Cgd":
        return cgd.Cgd()
    elif name == "Lstm":
    	return lstm.Lstm()

    return None

def getNeuralNetworkModels():
    return __CATEGORY__
