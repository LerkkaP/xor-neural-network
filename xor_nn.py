import numpy as np

class XorNet:
    def __init__(self):
        self._weights_hidden = np.array([[1, 1], [1, 1]])
        self._bias_hidden = np.array([0, -1])

        self._weights_output = np.array([[1, -2]])
        self._bias_output = np.array([0])

        self._hidden_layer_output = None
        self._output_layer_output = None

    def _relu(self, x: np.ndarray):
        return np.maximum(0, x)

    def _forward_propagation(self, x: np.ndarray):
        z_hidden = np.dot(x, self._weights_hidden.T) + self._bias_hidden
        self._hidden_layer_output = self._relu(z_hidden)

        z_output = np.dot(self._hidden_layer_output, self._weights_output.T) + self._bias_output
        self._output_layer_output = self._relu(z_output)
    
        return self._output_layer_output

    def _backpropagation(self):
        # TODO
        pass

    def train_step(self, x: np.ndarray):
        output = self._forward_propagation(x)
        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self._forward_propagation(x)
        return prediction