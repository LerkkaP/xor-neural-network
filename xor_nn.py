import numpy as np

class XorNet:
    def __init__(self):
        self._x = None

        self._w_hidden = np.random.rand(2, 2) 
        self._b_hidden = np.random.rand(2)

        self._w_output = np.random.rand(1, 2) 
        self._b_output = np.random.rand(1)

        self._z_hidden = None # Linear part of hidden layer
        self._z_output = None # Linear part of output layer

        self._h = None # Output of hidden layer
        self._o = None # Output of output layer

    def _forward_propagation(self, x: np.ndarray):
        self._x = x
        self._z_hidden = np.dot(x, self._w_hidden.T) + self._b_hidden
        self._h = self._relu(self._z_hidden)

        self._z_output = np.dot(self._h, self._w_output.T) + self._b_output
        self._o = self._relu(self._z_output)
    
        return self._o

    def _backpropagation(self, y_true):
        # Gradients for the output layer
        dc_do = self._cost_function_prime(y_true) # ∂c/∂o
        do_dz = self._relu_prime(self._z_output) # ∂o/∂z
        dc_dz = dc_do * do_dz # ∂c/∂o · ∂o/∂z

        dc_dw = np.dot(dc_dz.T, self._h) # ∂c/∂w = ∂c/∂o · ∂o/∂z · ∂z/∂w 
        dc_db = np.sum(dc_dz, axis=0) # ∂c/∂b

        # Gradients for the hidden layer
        dc_dh = np.dot(dc_dz, self._w_output)
        dh_dz_hidden = self._relu_prime(self._z_hidden)
        dc_dz_hidden = dc_dh * dh_dz_hidden
        
        dc_dw_hidden = np.dot(dc_dz_hidden.T, self._x)
        dc_db_hidden = np.sum(dc_dz_hidden, axis=0)      

        self._update_weights(dc_dw, dc_db, dc_dw_hidden, dc_db_hidden)

    def _relu(self, x: np.ndarray):
        return np.maximum(0, x)

    def _relu_prime(self, x: np.ndarray):
        return (x > 0).astype(float)

    def _cost_function(self, y_true: np.ndarray):
        cost = np.mean((y_true - self._o) ** 2)
        return cost
    
    def _cost_function_prime(self, y_true: np.ndarray):
        cost_prime = 2 * (self._o - y_true) / y_true.shape[0]
        return cost_prime
    
    def _update_weights(self, dc_dw, dc_db, dc_dw_hidden, dc_db_hidden, step_size = 0.1):
        self._w_output -= step_size * dc_dw
        self._b_output -= step_size * dc_db
        self._w_hidden -= step_size * dc_dw_hidden
        self._b_hidden -= step_size * dc_db_hidden

    def train_step(self, x: np.ndarray, y_true: np.ndarray):
        output = self._forward_propagation(x)
        self._backpropagation(y_true)  
        cost = self._cost_function(y_true)
        return (output, cost)

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self._forward_propagation(x)
        return prediction