import tensorflow as tf
import random

class MLP():
    def __init__(self):
        self._numOfLayers = 0
        self._neurons = list()
        self._activations = list()
        self._dropout = list()
        self._variables = dict()
        self._inputDim = 0
        self._outputDim = 2

        self._X_placeholder =  None
        self._y_true =  None
        self._dropout_list_placeholder = None
        self._y_pred = None

    def setNeurons(n):
        self._neurons = n
        self._numOfLayers = len(n)

    def setAct(a):
        self._activations = a

    def setDropout(p):
        self._dropout = p

    def setInputOutputDim(i, o):
        self._inputDim = i
        self._outputDim = o

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

    def _setPlaceholders(self):
        self._X_placeholder =  tf.placeholder(tf.float64, shape=[None, self._inputDim])
        self._y_true =  tf.placeholder(tf.float64, shape=[None, self._outputDim])
        self._dropout_list_placeholder = [tf.placeholder(tf.float64)]*len(self._dropout)

    def _activations(a, input):
        if a == "elu":
            return tf.nn.elu(input)
        elif a == "sigmoid":
            return tf.sigmoid(input)
        elif a == "tanh":
            return tf.tanh(input)
        elif a == "relu6":
            return tf.nn.relu6(input)
        elif a == "relu":
            return tf.nn.relu(input)
        else:
            print("Currently only supports elu, sigmoid, tanh, relu6 and relu. ReLu is chosen instead.")
            return tf.nn.relu(input)

    def _buildCompGraph(self):
        self._setPlaceholders()
        input = self._X_placeholder

        for i in range(self._numOfLayers):
            input = tf.add( tf.matmul(input, self._variables["W{}".format(i)]), self._variables["b{}".format(i)] )
            input = self._activations(self._activations[i], input)
            input = tf.nn.dropout(input, keep_prob=self._dropout_list_placeholder[i])

        self._y_pred = input

    def shuffledIndices(total_samples, seed):
        random.seed(seed)
        return random.sample(range(total_samples), total_samples)

    def buildModel( neurons, activations, dropout, inputDim, outputDim):
        assert(len(neurons) == len(dropout)), "Length of neurons and dropout lists are different!"
        assert(len(activations) == len(neurons)), "Length of activations and neurons list are different!"

        self.setNeurons(neurons)
        self.setAct(activations)
        self.setDropout(dropout)
        self.setInputOutputDim(inputDim, outputDim)

        self._setVariables()
        self._buildCompGraph()

    def train(X, y, lr=1e-3, num_epochs=100, batch_size=128, seed=1):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self._y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
        total_samples = int(X.shape[0])

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                indices = self.shuffledIndices(total_samples, seed)
                train_index = 0
                batch_index = 0
                num_batches = int(total_samples/batch_size)

                while batch_index <= num_batches:
                        X_batch = X[train_index:train_index+batch_size]
                        train += batch_size
                    batch_index += 1
                seed += 1
