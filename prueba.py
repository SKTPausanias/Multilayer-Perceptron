import pandas as pd
import sys
import numpy as np
from sklearn.metrics import accuracy_score
class MyNeuralNetwork():
    def __init__(self, input_features, alpha=0.001, epochs=1000):
        """
        Constructor.
        Args:
        X: has to be an numpy.ndarray, a matrix of dimension n * m.
        y: has to be an numpy.ndarray, a vector of dimension n * 1.
        alpha: has to be a float.
        epochs: has to be an int.
        Raises:
        This function should not raise any Exception.
        """
        self.layers = [input_features, input_features//2, input_features//2, 1]
        self.alpha = alpha
        self.epochs = epochs
        self.params = {}

        print("architecture", self.layers)
    
    def init_params(self):
        """
        Initialize the parameters of the model.
        """
        np.random.seed(1)
        self.params['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2],)
        self.params['W3'] = np.random.randn(self.layers[2], self.layers[3])
        self.params['b3'] = np.random.randn(self.layers[3],)

    def fit(self, X, y):
        """
        Train the model.
        Args:
        x: has to be an numpy.ndarray, a matrix of dimension n * m.
        y: has to be an numpy.ndarray, a vector of dimension n * 1.
        Raises:
        This function should not raise any Exception.
        """
        self.init_params()
        for i in range(self.epochs):
            yhat, loss = self.forward_prop(X, y)
            self.backward_prop(X, yhat, y)
            print("Epoch: {}, Loss: {}".format(i, loss))

    def forward_prop(self, X, y):
        """
        Forward propagation.
        Args:
        x: has to be an numpy.ndarray, a matrix of dimension n * m.
        Raises:
        This function should not raise any Exception.
        """
        z1 =  X.dot(self.params['W1']) + self.params['b1']
        a1 = self.relu(z1)
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        a2 = self.relu(z2)
        z3 = a2.dot(self.params['W3']) + self.params['b3']
        y_hat = self.sigmoid(z3)
        loss = self.entropy_loss(y, y_hat)

        self.params['z1'] = z1
        self.params['a1'] = a1
        self.params['z2'] = z2
        self.params['a2'] = a2
        self.params['z3'] = z3
        return y_hat, loss
    
    def backward_prop(self, X, yhat, y):
        """
        Backward propagation.
        Args:
        x: has to be an numpy.ndarray, a matrix of dimension n * m.
        y: has to be an numpy.ndarray, a vector of dimension n * 1.
        Raises:
        This function should not raise any Exception.
        """
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(y, self.eta(yhat))
        dl_wrt_sig = yhat * (yhat_inv)
        dl_wrt_z3 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_a2 = dl_wrt_z3.dot(self.params['W3'].T)
        dl_wrt_w3 = self.params['a2'].T.dot(dl_wrt_z3)
        dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0)

        dl_wrt_z2 = dl_wrt_a2 * self.dRelu(self.params['z2'])
        dl_wrt_a1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['a1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_a1 * self.dRelu(self.params['z1'])
        dl_wrt_w1 = X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        self.params['W3'] -= (self.alpha * dl_wrt_w3)
        self.params['b3'] -= (self.alpha * dl_wrt_b3)
        self.params['W2'] -= (self.alpha * dl_wrt_w2)
        self.params['b2'] -= (self.alpha * dl_wrt_b2)
        self.params['W1'] -= (self.alpha * dl_wrt_w1)
        self.params['b1'] -= (self.alpha * dl_wrt_b1)
    
    def predict(self, x, y):
        """
        Prediction of the class of a single sample.
        Args:
        x: has to be an numpy.ndarray, a single sample.
        Returns:
        The class of the sample.
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        if x.size == 0 or x is None:
            return None
        y_hat, loss = self.forward_prop(x, y)
        rounded = np.round(y_hat)
        return rounded

    def eta(self, x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)


    def sigmoid(self,Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1/(1+np.exp(-Z))

    def entropy_loss(self,y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = self.eta(yhat_inv) 
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss
    

    def soft_max(self, x):
        """
        Compute the softmax of a vector.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        The softmax value as a numpy.ndarray.
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        if x.size == 0 or x is None:
            return None
        return np.exp(x) / np.sum(np.exp(x))
    
    def relu(self,Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        return np.maximum(0,Z)
    
    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

if __name__  == "__main__":
    #take csv from /datasets
    df = pd.read_csv('datasets/data.csv', header=None)
    if (df is None):
        print("Error: could not read csv file")
        sys.exit(1)
    
    #remove first column
    df.drop(df.columns[0], axis=1, inplace=True)
    #map values in first column, giving 1 if tumor is Malignant (M) and 0 if Benign (B)
    df[1] = df[1].map({'M': 1, 'B': 0})

    #split datafarme into training and testing data
    train_df = df.sample(frac=0.6, random_state=200)
    test_df = df.drop(train_df.index)

    y_train = np.array(train_df.iloc[:, 0])
    y_test = np.array(test_df.iloc[:, 0])

    #drop first column from training 
    train_df.drop(train_df.columns[0], axis=1, inplace=True)
    train_df = (train_df - train_df.mean()) / train_df.std()

    test_df.drop(test_df.columns[0], axis=1, inplace=True)
    test_df = (test_df - test_df.mean()) / test_df.std()

    X = np.array(train_df)
    X_test = np.array(test_df)

    y = np.array(y_train)
    y_test = np.array(y_test)

    y = y.reshape(len(y), 1)
    y_test = y_test.reshape(len(y_test), 1)
    #print("X: ", X.shape)
    #print("y: ", y.shape)

    #initialize neural network with 2 hidden layers
    print(X.shape[1])
    nn = MyNeuralNetwork(X.shape[1], alpha=0.001, epochs=100)
    nn.fit(X, y)

    #predict on test data
    yhat = nn.predict(X_test, y_test)
    #print("yhat: ", yhat)
    print("y_test: ", y_test.shape)
    print("Accuracy: ", accuracy_score(y_test, yhat))

