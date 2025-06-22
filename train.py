import numpy as np
from xor_nn import XorNet
from visualizer import Visualizer

def main():
    training_data = np.array([(0, 0), (1, 1),
                            (0, 1), (1, 0)])
    model = XorNet()
    visualizer = Visualizer()
    
if __name__ == "__main__":
    main()


