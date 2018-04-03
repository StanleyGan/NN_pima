import tensorflow as tf
import random
import itertools
import copy
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
matplotlib.style.use('ggplot')

'''
A multilayer perceptron class
numOfLayers: number of hidden layers of the MLP, an int
neurons: number of neurons in each hidden layer, a list of int
activations: activation functions in each hidden layer, a list of strings.
            Currently only accepts elu, sigmoid, tanh, relu6 and relu
dropout: Keep_prob rate for dropout layers, a list of floats
variables: Tensorflow trainable variables, weights and biases. A dictionary of tf.Variable
inputDim: An int, number of features
outputDim: An int, number of classes
X_placeholder: A tf.placeholder which will contain the input of the MLP
y_placeholder: A tf.placeholder which will contain the actual label (used for
                cost calculation to train the network and accuracy calculation)
dropout_list_placeholder: A list of tf.placeholder which will contain Keep_prob
                            rate for each dropout layer
y_pred: Predicted output value (not labels i.e. before softmax)
trained_variables: Values for each weight and biases, a dictionary. Can be used
                    for future predictions

'''
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
        self.trained_variables = dict()

    #Create new weight matrix
    #Input: shape i.e. in form of [x,y]
    def _setWeights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    #Create new bias
    #Input: shape i.e. in form of [x]
    def _setBias(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    #Create variables and set numOfLayers, neurons, inputDim
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

    #Create placeholders
    def _setPlaceholders(self):
        self.X_placeholder =  tf.placeholder(tf.float32, shape=[None, self._inputDim])
        self.y_placeholder =  tf.placeholder(tf.float32, shape=[None, self._outputDim])
        self.dropout_list_placeholder = [tf.placeholder(tf.float32) for _ in xrange(len(self._dropout))]

    #Add activation functions
    #Input: a, a string
    #       input, a Tensor
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

    #Build the computational graph
    #Input: variables, a dictionary which contains actual values for self._variables
    def _buildCompGraph(self, variables):
        assert(set(self._variables.keys()) == set(variables.keys())), "Given variables do not have matching keys"
        self._setPlaceholders()
        input = self.X_placeholder

        for i in range(self.returnNumOfLayers()):
            input = tf.add( tf.matmul(input, variables["W{}".format(i+1)]), variables["b{}".format(i+1)] )
            input = self._applyActivations(self._activations[i], input)
            input = tf.nn.dropout(input, keep_prob=self.dropout_list_placeholder[i])

        input = tf.add( tf.matmul(input, variables["W{}".format(self.returnNumOfLayers()+1)]), variables["b{}".format(self.returnNumOfLayers()+1)] )
        self.y_pred = input

    #Set self._neurons
    #Input: n, a list of int
    def setNeurons(self,n):
        self._neurons = n
        self._numOfLayers = len(n)

    #Return number of hidden layers
    def returnNumOfLayers(self):
        return self._numOfLayers

    #Set activation functions
    #Input: a, a list of strings
    def setAct(self,a):
        self._activations = a

    #Set dropout rates
    #Input: p, a list of floats
    def setDropout(self,p):
        self._dropout = p

    #Set input and output dimension
    #Input: i, an int
    #       o, an int
    def setInputOutputDim(self,i, o):
        self._inputDim = i
        self._outputDim = o

    #Shuffle indices for training in batches
    #Input: total_samples, an int indicating number of samples
    #        seed, an int
    def shuffledIndices(self,total_samples, seed):
        random.seed(seed)
        return random.sample(range(total_samples), total_samples)

    #Build the MLP model by specifying the configurations and build the computational
    #graph.
    def buildModel(self, neurons, activations, dropout, inputDim, outputDim):
        assert(len(neurons) == len(dropout)), "Length of neurons and dropout lists are different!"
        assert(len(activations) == len(neurons)), "Length of activations and neurons list are different!"

        self.setNeurons(neurons)
        self.setAct(activations)
        self.setDropout(dropout)
        self.setInputOutputDim(inputDim, outputDim)

        self._setVariables()
        self._buildCompGraph(self._variables)

    '''
    Train the MLP
    Input: X, training set. A 2D array
            y, training labels. A 2D array
            X_test, test set. A 2D array
            y_test, test labels. A 2D array
            lr, learning rate for AdamOptimizer. A float.
            num_epochs
            batch_size
            seed
            printResults, indicator to print cost and graph. A boolean.
    '''
    def train(self, X, y, X_test, y_test, lr=1e-3, num_epochs=100, batch_size=128, seed=1, printResults=True, returnResults=False):
        #Set seed for reproducibility
        tf.set_random_seed(seed)
        shuffling_seed = copy.deepcopy(seed)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y_placeholder))
        accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(self.y_pred,1), tf.argmax(self.y_placeholder,1)), tf.float32) )
        optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
        total_samples = int(X.shape[0])

        #Intialize variables
        init = tf.global_variables_initializer()
        training_loss = list()  #for plotting

        #Start a tensorflow session
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                #Obtain a randomly shuffled indices for batch training
                indices = self.shuffledIndices(total_samples, shuffling_seed)
                train_index = 0

                #Train on the randomly shuffled indices in a sequential manner
                #Start a new shuffling if reaches the end of sequence
                while train_index+batch_size <= total_samples:
                    #Obtain batches
                    X_batch = X[train_index:train_index+batch_size,:]
                    y_batch = y[train_index:train_index+batch_size,:]
                    #Set up feed_dict for training
                    feed_dict_train = {ph:do for (ph, do) in itertools.izip(self.dropout_list_placeholder, self._dropout)}
                    feed_dict_train[self.X_placeholder] = X_batch
                    feed_dict_train[self.y_placeholder] = y_batch
                    sess.run(optimizer, feed_dict=feed_dict_train)
                    train_index += batch_size

                #If train_index hasn't reached the end of the sequence
                if train_index < total_samples:
                    X_batch = X[train_index:,:]
                    y_batch = y[train_index:,:]
                    feed_dict_train = {ph:do for (ph, do) in itertools.izip(self.dropout_list_placeholder, self._dropout)}
                    feed_dict_train[self.X_placeholder] = X_batch
                    feed_dict_train[self.y_placeholder] = y_batch
                    sess.run(optimizer, feed_dict=feed_dict_train)

                #Calculate cost and accuracy for report
                feed_dict_train = {ph:do for (ph, do) in itertools.izip(self.dropout_list_placeholder, self._dropout)}
                feed_dict_train[self.X_placeholder] = X_batch
                feed_dict_train[self.y_placeholder] = y_batch
                train_cost = sess.run(cost, feed_dict=feed_dict_train)
                training_loss.append(train_cost)

                feed_dict_test = {ph:1.0 for ph in self.dropout_list_placeholder}
                feed_dict_test[self.X_placeholder] = X_test
                feed_dict_test[self.y_placeholder] = y_test
                test_acc = sess.run(accuracy, feed_dict=feed_dict_test )

                #Print cost
                if epoch % 10 == 0 and printResults==True:
                    print("Epoch {0} : Training loss: {1}, \n test accuracy : {2}\n".format(epoch+1, train_cost, test_acc))

                #In order to produce different shuffling
                shuffling_seed += 1

                #Store results
                results=dict()
                if epoch == num_epochs-1 and returnResults == True:
                    results["train_loss"] = training_loss
                    results["test_acc"] = test_acc

            #Plot training loss graph
            if printResults== True:
                plt.plot(training_loss)
                plt.show()

            #save trained variables as separate dictionary with actual numbers
            self.trained_variables = sess.run(self._variables)

            #Return results if returnResults is True
            if returnResults == True:
                return results
            else:
                pass

    #Predict on a set of features and returns class labels in the form of
    # 2D array i.e. [[0,1] [1.0]]. It uses trained variables to predict.
    #Input: X, data set. A 2D array
    #        return_prob, True if user wants to return the probability value
    def predict(self,X, return_prob=False, seed=1):
        assert(len(self.trained_variables) != 0), "Please train the model before predicting!"
        tf.set_random_seed(seed)
        self._buildCompGraph(self.trained_variables)

        res=dict()

        with tf.Session() as sess:
            sess = tf.Session()
            feed_dict = {ph:1.0 for ph in self.dropout_list_placeholder}
            feed_dict[self.X_placeholder] = X

            y_pred = sess.run(self.y_pred, feed_dict=feed_dict)
            #labels
            y_pred_cls = sess.run( tf.round(tf.nn.softmax(y_pred)) )
            #probability
            y_pred = sess.run( tf.nn.softmax(y_pred) )

            res["y_pred_cls"] = y_pred_cls

            if return_prob == True:
                res["y_pred_prob"]= y_pred

        return res
