import numpy as np

class BatchGDLinearRegression:
    def __init__(self, epochs=50, learning_rate=1e-3):
        self.coef_, self.intercept_ = None, None
        self.eta = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, y_train):

        y_train = y_train.reshape(-1, 1)
        n_samples, n_features = X_train.shape

        self.coef_ = np.random.rand(n_features, 1)
        self.intercept_ = np.random.rand(1, 1)

        for epoch in range(self.epochs):
            y_hat = X_train @ self.coef_ + self.intercept_
            error = y_train - y_hat

            slope_intercept = -2 * np.mean(error)
            slope_coef = -2 * (X_train.T @ error) / n_samples

            self.coef_ -= self.eta * slope_coef
            self.intercept_ -= self.eta * slope_intercept
    
    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_