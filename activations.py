import numpy as np

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self(x)
        return s * (1 - s)

class ReLU():
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)
