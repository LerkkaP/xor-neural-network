import numpy as np
from activations import Sigmoid, ReLU
from costs import MSE, BCE
from xor_nn import XorNet
from visualizer import Visualizer

def main():
    training_data = np.array([[0, 0], [1, 0],[0, 1], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])

    model = XorNet(Sigmoid(), BCE())
    visualizer = Visualizer()

    costs = []
    for _ in range(100000):
        output, cost = model.train_step(training_data, labels)
        costs.append(cost)
    print(model.predict(training_data))

    visualizer.plot_loss(costs)

if __name__ == "__main__":
    main()


