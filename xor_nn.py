import numpy as np

class XorNet:
    def __init__(self):
        self._weights_hidden = np.random.rand(2, 2)
        self._bias_hidden = np.random.rand(2)

        self._weights_output = np.random.rand(1, 2)
        self._bias_output = np.random.rand(1)

        self._hidden_layer = None
        self._output_layer = None

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward_propagation(self, x):
        z_hidden = np.dot(self._weights_hidden, x) + self._bias_hidden
        self._hidden_layer = self._relu(z_hidden)

        z_output = np.dot(self._weights_output, self._hidden_layer) + self._bias_output
        self._output_layer = self._relu(z_output)
    
        return self._output_layer

    def _backpropagation(self):
        # TODO
        pass

    def train_step(self):
        # TODO
        pass

    def predict(self):
        # TODO 
        pass