import numpy as np

class L2SigmoidPerceptron:
    def __init__(self, epochs=50, learning_rate=1e-1, lambda_=1):
        self.weights = None

        self.epochs = epochs
        self.eta = learning_rate
        self.lambda_ = lambda_

    def __sigmoid(self, z):
        return 1 / ( 1 + (np.exp(-z)) )

    def fit(self, X_train, y_train):
        X_train = np.c_[ (np.ones( (X_train.shape[0], 1) )) ,X_train ]
        y_train = y_train.reshape(-1, 1)

        n_samples, n_features = X_train.shape
        self.weights = np.random.rand(n_features, 1)

        for epoch in range(self.epochs):
            y_hat = self.__sigmoid( X_train @ self.weights ).reshape(-1, 1)
            gradients = ( X_train.T @ (y_train - y_hat) ) / n_samples
            reg = (self.lambda_ / (n_samples)) * (self.weights)
            reg[0] = 0  # prevent bias from regularization

            self.weights = self.weights + self.eta *( gradients + reg )  # regularized term added here

    def predict(self, X_test):
        X_test = np.c_[ (np.ones((X_test.shape[0], 1))), X_test ]
        probs = self.__sigmoid(( X_test @ self.weights ))
        return (probs >= 0.5).astype(np.int32)