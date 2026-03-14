"""
plotter.py

Plot training metrics.
"""

import os
import sys
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
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()
        # In notebook/Colab runs started from scripts, explicitly display saved image.
        if save_path and ("ipykernel" in sys.modules or "google.colab" in sys.modules):
            try:
                from IPython.display import Image, display
                display(Image(filename=save_path))
            except Exception:
                pass
