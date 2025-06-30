import numpy as np
from xor_nn import XorNet
from visualizer import Visualizer

def main():
    training_data = np.array([[0, 0], [1, 0],[0, 1], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])

    model = XorNet()
    visualizer = Visualizer()
    costs = []
    for i in range(10000):
        output, cost = model.train_step(training_data, labels)
        if i % 1000 == 0:
            print(f"Step {i}, cost: {cost}")
        costs.append(cost)
    print(model.predict(training_data))

if __name__ == "__main__":
    main()


