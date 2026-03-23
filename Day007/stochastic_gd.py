import numpy as np

class StochasticGDLinearRegression:
    def __init__(self, epochs=50, learning_rate=1e-3):
        self.coef_, self.intercept_ = None, None
        
        self.epochs = epochs
        self.eta = learning_rate
    
    def fit(self, X_train, y_train):

        y_train = y_train.reshape(-1, 1)
        n_samples, n_features = X_train.shape

        self.coef_ = np.random.rand(n_features, 1)
        self.intercept_ = np.random.rand(1, 1)

        for epoch in range(self.epochs):
            for idx in range(n_samples):
                random_idx = np.random.randint(0, n_samples)

                Xi = X_train[random_idx: random_idx+1]
                yi = y_train[random_idx: random_idx+1].reshape(1, 1)

                y_hat = Xi @ self.coef_ + self.intercept_
                error = yi - y_hat

                slope_intercept = -2 * (error)
                slope_coef = -2 * (Xi.T @ error)

                self.coef_ -= self.eta * slope_coef
                self.intercept_ -= self.eta * slope_intercept
    
    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_