import numpy as np

class L2RegularizationOLS:
    def __init__(self, alpha=1,):
        self.alpha = alpha
        self.intercept_, self.coef_ = None, None

    def fit(self, X_train, y_train):

        X_train = np.c_[(np.ones((X_train.shape[0], 1))), X_train]
        I = np.identity(X_train.shape[1])
        I[0, 0] = 0     # to prevent regularization on intercept
        betas = (np.linalg.inv( X_train.T @ X_train + (self.alpha * I) )) @ X_train.T @ y_train

        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self, X_test):
        return X_test @ self.coef_ + self.intercept_