import numpy as np

class L2RegularizationBGD:
    def __init__(self, epochs=50, learning_rate=1e-3, alpha=1):
        self.coef_, self.intercept_ = None, None
        
        self.alpha = alpha
        self.eta = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, y_train):

        X_train = np.c_[( np.ones((X_train.shape[0], 1)) ), X_train]
        y_train = y_train.reshape(-1, 1)

        n_samples, n_features = X_train.shape
        
        # first set gradient to random values
        betas = np.random.rand(n_features, 1)

        # run loop for given no. of epochs
        for epoch in range(self.epochs):
            predictions = X_train @ betas
            errors = y_train - predictions

            gradients = (X_train. T @ errors) / n_samples

            # regularization
            reg = self.alpha * betas
            reg[0] = 0      # don't regularize the intercept term.

            gradients += reg

            betas -= 2 * self.eta * gradients
        
        # update coef_ and intercept_ values
        self.intercept_ = betas[0]
        self.coef_ = betas[1]
    
    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_