import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_loss(costs):
        plt.figure(figsize=(8, 5))
        plt.plot(costs, label='Loss')
        plt.title('Training Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig("loss_plot.png")
        plt.close()

    @staticmethod
    def plot_decision_boundary(f, training_data, labels):
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = f(grid).reshape(xx.shape)

        plt.figure(figsize=(6, 6))
        plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=['#FFAAAA', '#AAAAFF'], alpha=0.6)
        plt.contour(xx, yy, zz, levels=[0.5], colors='black')

        plt.scatter(training_data[:, 0], training_data[:, 1], c=labels.ravel(),
                    cmap=plt.cm.RdBu, edgecolors='k', s=100)
        plt.title("XOR Decision Boundary")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("decision_boundary.png")
        plt.close()