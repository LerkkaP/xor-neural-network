import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_loss():
        # TODO
        pass

    @staticmethod
    def plot_decision_boundary():
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]

        fig, ax = plt.subplots()
        ax.scatter(x, y, color='red')

        plt.title("Decision boundary")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()

        plt.show()