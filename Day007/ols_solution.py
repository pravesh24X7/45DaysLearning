import numpy as np

class OLSLinearRegression:
    def __init__(self):
        self.__X = None
        self.__y = None

        self.intercept_ = None
        self.coef_ = None

    def fit(self, X_train, y_train):
        
        self.__X = np.c_[(np.ones((X_train.shape[0], 1))), X_train]
        self.__y = y_train

        betas = np.linalg.inv( self.__X.T @ self.__X ) @ self.__X.T @ self.__y        
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
