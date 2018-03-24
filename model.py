import tensorflow as tf
import random
import itertools
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

class MLP():
    def __init__(self):
        self._numOfLayers = 0
        self._neurons = list()
        self._activations = list()
        self._dropout = list()
        self._variables = dict()
        self._inputDim = 0
        self._outputDim = 2

        self.X_placeholder =  None
        self.y_placeholder =  None
        self.dropout_list_placeholder = None
        self.y_pred = None

    def _setWeights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def _setBias(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def _setVariables(self):
        numOfLayers = self.returnNumOfLayers()
        neurons = self._neurons
        inputDim = self._inputDim

        for i in range(numOfLayers):
            self._variables["W{}".format(i+1)] = self._setWeights([inputDim, neurons[i]])
            self._variables["b{}".format(i+1)] = self._setBias([neurons[i]])
            inputDim = neurons[i]

        self._variables["W{}".format(numOfLayers+1)] = self._setWeights([neurons[numOfLayers-1], self._outputDim])
        self._variables["b{}".format(numOfLayers+1)] = self._setBias([self._outputDim])

    def _setPlaceholders(self):
        self.X_placeholder =  tf.placeholder(tf.float32, shape=[None, self._inputDim])
        self.y_placeholder =  tf.placeholder(tf.float32, shape=[None, self._outputDim])
        self.dropout_list_placeholder = [tf.placeholder(tf.float32) for _ in xrange(len(self._dropout))]

    def _applyActivations(self,a, input):
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
        input = self.X_placeholder

        for i in range(self.returnNumOfLayers()):
            input = tf.add( tf.matmul(input, self._variables["W{}".format(i+1)]), self._variables["b{}".format(i+1)] )
            input = self._applyActivations(self._activations[i], input)
            input = tf.nn.dropout(input, keep_prob=self.dropout_list_placeholder[i])

        input = tf.add( tf.matmul(input, self._variables["W{}".format(self.returnNumOfLayers()+1)]), self._variables["b{}".format(self.returnNumOfLayers()+1)] )
        self.y_pred = input

    def setNeurons(self,n):
        self._neurons = n
        self._numOfLayers = len(n)

    def returnNumOfLayers(self):
        return self._numOfLayers

    def setAct(self,a):
        self._activations = a

    def setDropout(self,p):
        self._dropout = p

    def setInputOutputDim(self,i, o):
        self._inputDim = i
        self._outputDim = o

    def shuffledIndices(self,total_samples, seed):
        random.seed(seed)
        return random.sample(range(total_samples), total_samples)

    def buildModel(self, neurons, activations, dropout, inputDim, outputDim):
        assert(len(neurons) == len(dropout)), "Length of neurons and dropout lists are different!"
        assert(len(activations) == len(neurons)), "Length of activations and neurons list are different!"

        self.setNeurons(neurons)
        self.setAct(activations)
        self.setDropout(dropout)
        self.setInputOutputDim(inputDim, outputDim)

        self._setVariables()
        self._buildCompGraph()

    def train(self, X, y, X_test, y_test, lr=1e-3, num_epochs=100, batch_size=128, seed=1):
        # X = tf.cast(X, tf.float32)
        # y = tf.cast(y, tf.float32)
        # X_test = tf.cast(X_test, tf.float32)
        # y_test = tf.cast(y_test, tf.float32)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y_placeholder))
        accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(self.y_pred,1), tf.argmax(self.y_placeholder,1)), tf.float32) )
        optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
        total_samples = int(X.shape[0])

        init = tf.global_variables_initializer()
        training_loss = list()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                indices = self.shuffledIndices(total_samples, seed)
                train_index = 0

                while train_index+batch_size <= total_samples:
                    X_batch = X[train_index:train_index+batch_size,:]
                    y_batch = y[train_index:train_index+batch_size,:]
                    feed_dict_train = {ph:do for (ph, do) in itertools.izip(self.dropout_list_placeholder, self._dropout)}
                    feed_dict_train[self.X_placeholder] = X_batch
                    feed_dict_train[self.y_placeholder] = y_batch
                    sess.run(optimizer, feed_dict=feed_dict_train)
                    train_index += batch_size

                if train_index < total_samples:
                    X_batch = X[train_index:,:]
                    y_batch = y[train_index:,:]
                    feed_dict_train = {ph:do for (ph, do) in itertools.izip(self.dropout_list_placeholder, self._dropout)}
                    feed_dict_train[self.X_placeholder] = X_batch
                    feed_dict_train[self.y_placeholder] = y_batch
                    sess.run(optimizer, feed_dict=feed_dict_train)

                feed_dict_train = {ph:do for (ph, do) in itertools.izip(self.dropout_list_placeholder, self._dropout)}
                feed_dict_train[self.X_placeholder] = X_batch
                feed_dict_train[self.y_placeholder] = y_batch
                train_cost = sess.run(cost, feed_dict=feed_dict_train)
                training_loss.append(train_cost)

                feed_dict_test = {ph:1.0 for ph in self.dropout_list_placeholder}
                feed_dict_test[self.X_placeholder] = X_test
                feed_dict_test[self.y_placeholder] = y_test
                test_acc = sess.run(accuracy, feed_dict=feed_dict_test )

                if epoch % 10 == 0:
                    print("Epoch {0} : Training loss: {1}, \n test accuracy : {2}\n".format(epoch+1, train_cost, test_acc))
                seed += 1

            #Plot training loss graph
            plt.plot(training_loss)
            plt.show()

    def predict(self,X):
        sess = tf.Session()
        y_pred = sess.run(self.y_pred, feed_dict={X:X, self.dropout_list_placeholder:[1]*self.returnNumOfLayers()})

        return y_pred
