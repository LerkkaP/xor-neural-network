import numpy as np
from xor_nn import XorNet
from visualizer import Visualizer

def main():
    training_data = np.array([[0, 0], [1, 0],[0, 1], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])

    model = XorNet()
    visualizer = Visualizer()
    array = []
    for _ in range(10000):
        output, cost = model.train_step(training_data, labels)
        array.append((output, cost))
    print(model.predict(training_data))

if __name__ == "__main__":
    main()


