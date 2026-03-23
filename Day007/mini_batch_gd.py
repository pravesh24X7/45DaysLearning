import numpy as np

class MiniBatchGDLinearRegression:
    def __init__(self, epochs=50, learning_rate=1e-3, batch_size=32):
        self.batch_size = batch_size
        self.eta = learning_rate
        self.epochs = epochs

        self.coef_, self.intercept_ = None, None
    
    def fit(self, X_train, y_train):

        y_train = y_train.reshape(-1, 1)
        n_samples, n_features = X_train.shape

        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        self.coef_ = np.random.rand(n_features, 1)
        self.intercept_ = np.random.rand(1, 1)

        for epoch in range(self.epochs):
            for i in range(self.batch_size):

                Xi = X_train[i: i+self.batch_size]
                yi = y_train[i: i+self.batch_size]

                y_hat = Xi @ self.coef_ + self.intercept_
                error = yi - y_hat

                slope_intercept = -2 * np.mean(error)
                slope_coef = -2 * (Xi.T @ error) / Xi.shape[0]

                self.coef_ -= self.eta * slope_coef
                self.intercept_ -= self.eta * slope_intercept
    
    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_