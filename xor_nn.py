import numpy as np

class XorNet:
    def __init__(self):
        self._weights_hidden = np.random.rand(2, 2)
        self._bias_hidden = np.random.rand(2)

        self._weights_output = np.random.rand(1, 2)
        self._bias_output = np.random.rand(1)

        self._hidden_layer_output = None
        self._output_layer_output = None

        self._cost = None
        self._cost_prime = None

    def _relu(self, x: np.ndarray):
        return np.maximum(0, x)

    def _relu_prime(self, x: np.ndarray):
        return (x > 0).astype(float)

    def _forward_propagation(self, x: np.ndarray):
        z_hidden = np.dot(x, self._weights_hidden.T) + self._bias_hidden
        self._hidden_layer_output = self._relu(z_hidden)

        z_output = np.dot(self._hidden_layer_output, self._weights_output.T) + self._bias_output
        self._output_layer_output = self._relu(z_output)
    
        return self._output_layer_output

    def _backpropagation(self, y_true):
        # Gradients for the output layer
        dc_dy_pred = self._cost_function_prime(y_true, self._hidden_layer_output)
        dy_pred_dz = self._relu_prime(self._hidden_layer_output)
        dc_dz = dc_dy_pred * dy_pred_dz
        dc_dw = np.dot(dc_dz.T, self._hidden_layer_output)


    def _cost_function(self, y_true: np.ndarray, y_pred: np.ndarray):
        cost = np.mean((y_true - y_pred) ** 2)
        self._cost = cost
        return cost
    
    def _cost_function_prime(self, y_true: np.ndarray, y_pred: np.ndarray):
        cost_prime = np.mean(-2 * (y_true - y_pred))
        self._cost_prime = cost_prime
        return cost_prime

    def train_step(self, x: np.ndarray, y_true: np.ndarray):
        output = self._forward_propagation(x)
        cost = self._cost_function(y_true, output)
        return (output, cost)

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self._forward_propagation(x)
        return prediction