"""
plotter.py

Plot training metrics.
"""

import matplotlib.pyplot as plt


class TrainingPlotter:

    @staticmethod
    def plot_loss(loss_history, save_path=None):
        """
        Plot training loss curve.
        """

        plt.figure()

        plt.plot(loss_history, label="Training Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()
