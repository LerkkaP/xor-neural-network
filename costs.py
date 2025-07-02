import numpy as np

class MSE():
    def __call__(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

class BCE:
    def __call__(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        return y_pred - y_true 
