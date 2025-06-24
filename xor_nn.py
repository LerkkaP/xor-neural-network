import numpy as np

class XorNet:
    def __init__(self):
        self._weights_hidden = np.random.rand(2, 2)
        self._bias_hidden = np.random.rand(2)

        self._weights_output = np.random.rand(1, 2)
        self._bias_output = np.random.rand(1)
        # TODO

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward_propagation(self):
        # TODO
        pass

    def _backpropagation(self):
        # TODO
        pass

    def train_step(self):
        # TODO
        pass