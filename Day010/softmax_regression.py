import numpy as np

class SoftmaxRegression:
    def __init__(self, epochs=50, learning_rate=1e-1):
        self.weights = None

        self.epochs = epochs
        self.eta = learning_rate

    def __softmax(self, z):
        exp_z = np.exp( z - np.max(z, axis=1, keepdims=True) )
        return exp_z / np.sum( exp_z, axis=1, keepdims=True )

    def fit(self, X_train, y_train, sparse=True):
        X_train = np.c_[ (np.ones( (X_train.shape[0], 1) )) ,X_train ]
        y_train = y_train.reshape(-1, 1)

        n_samples, n_features = X_train.shape
        n_classes = np.unique(y_train).size

        self.weights = np.random.randn( n_features, n_classes ) * 0.001

        for epoch in range(self.epochs):
            logits = X_train @ self.weights
            probs = self.__softmax(logits)

            if sparse:
                y_onehot = np.zeros((n_samples, n_classes))
                y_onehot[ np.arange(n_samples), y_train.flatten() ] = 1
            else:
                y_onehot = y_train

            gradient = X_train.T @ ( probs - y_onehot ) / n_samples
            self.weights = self.weights - self.eta * gradient

    def predict(self, X_test):
        X_test = np.c_[ (np.ones( (X_test.shape[0], 1) )), X_test ]
        logits = X_test @ self.weights
        probs = self.__softmax(logits)

        y_pred = np.argmax(probs, axis=1)
        return y_pred