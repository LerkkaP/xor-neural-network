import numpy as np
from xor_nn import XorNet
from visualizer import Visualizer

def main():
    training_data = np.array([[0, 0], [1, 1],
                            [0, 1], [1, 0]])
    model = XorNet()
    
    prediction = model.predict(training_data)
    print(prediction)

if __name__ == "__main__":
    main()


