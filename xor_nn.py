import numpy as np

class XorNet:
    def __init__(self):
        self._w_hidden = np.random.rand(2, 2)
        self._b_hidden = np.random.rand(2)

        self._w_output = np.random.rand(1, 2)
        self._b_output = np.random.rand(1)

        self._z_hidden = None # Linear part of hidden layer
        self._z_output = None # Linear part of output layer

        self._h = None # Output of hidden layer
        self._o = None # Output of output layer

    def _forward_propagation(self, x: np.ndarray):
        self._z_hidden = np.dot(x, self._w_hidden.T) + self._b_hidden
        self._h = self._relu(self._z_hidden)

        self._z_output = np.dot(self._h, self._w_output.T) + self._b_output
        self._o = self._relu(self._z_output)
    
        return self._o

    def _backpropagation(self, y_true):
        # Gradients for the output layer
        dc_dy_pred = self._cost_function_prime(y_true, self._h)
        dy_pred_dz = self._relu_prime(self._h)
        dc_dz = dc_dy_pred * dy_pred_dz
        dc_dw = np.dot(dc_dz.T, self._h)

    def _relu(self, x: np.ndarray):
        return np.maximum(0, x)

    def _relu_prime(self, x: np.ndarray):
        return (x > 0).astype(float)

    def _cost_function(self, y_true: np.ndarray):
        cost = np.mean((y_true - self._o) ** 2)
        return cost
    
    def _cost_function_prime(self, y_true: np.ndarray):
        cost_prime = np.mean(-2 * (y_true - self._o))
        return cost_prime

    def train_step(self, x: np.ndarray, y_true: np.ndarray):
        output = self._forward_propagation(x)
        cost = self._cost_function(y_true, output)
        return (output, cost)

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self._forward_propagation(x)
        return prediction