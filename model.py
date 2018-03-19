import tensorflow as tf

class MLP():
    def __init__(self):
        self._numOfLayers = 0
        self._neurons = list()
        self._activations = list()
        self._variables = dict()    
        self._inputDim = 0
        self._outputDim = 1

    def setNeurons(n):
        self._neurons = n
        self._numOfLayers = len(n)

    def setAct(a):
        self._activations = a

    def _setWeights(shape):
        return tf.Variable(tf.truncated_normal(shape, stdev=0.1))

    def _setBias(shape):
        return tf.Variable(tf.truncated_normal(shape, stdev=0.1))    
    
    def _setVariables(self):
        numOfLayers = self._numOfLayers
        neurons = self._neurons

        for i in range(len(numOfLayers)):
            self._variables["W{}".format(i+1)] = self._setWeights([self._inputDim, neurons[i]])
            self._variables["b{}".format(i+1)] = self._setBias([neurons[i]])

        self._variables["W{}".format(i+1)] = self._setWeights([neurons[i], self._outputDim])
        self._variables["b{}".format(i+1)] = self._setBias([self._outputDim])


    def _buildCompGraph(self):
        

    def setModel( neurons, activations, inputDim, outputDim):
        self.setNeurons(neurons)
        self.setAct(activations)
        self._inputDim = inputDim
        self._outputDim = outputDim

        self._setVariables()
        
    
