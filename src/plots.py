"""
=========================================================
plots.py
Generate Training Curves
=========================================================
"""

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def generate_plots():

    print("=" * 60)
    print("Generating Training Curves")
    print("=" * 60)

    history = pd.read_csv(
        config.RESULTS_PATH / "history.csv"
    )

    # ==========================================
    # Accuracy Plot
    # ==========================================

    plt.figure(figsize=(10,6))

    plt.plot(
        history["round"],
        history["train_accuracy"],
        marker="o",
        linewidth=2,
        label="Training Accuracy"
    )

    plt.plot(
        history["round"],
        history["validation_accuracy"],
        marker="s",
        linewidth=2,
        label="Validation Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Communication Round",
        fontsize=13
    )

    plt.ylabel(
        "Accuracy",
        fontsize=13
    )

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(fontsize=12)

    plt.tight_layout()

    accuracy_path = (
        config.RESULTS_PATH /
        "accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ==========================================
    # Loss Plot
    # ==========================================

    plt.figure(figsize=(10,6))

    plt.plot(
        history["round"],
        history["train_loss"],
        marker="o",
        linewidth=2,
        label="Training Loss"
    )

    plt.plot(
        history["round"],
        history["validation_loss"],
        marker="s",
        linewidth=2,
        label="Validation Loss"
    )

    plt.title(
        "Training and Validation Loss",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Communication Round",
        fontsize=13
    )

    plt.ylabel(
        "Loss",
        fontsize=13
    )

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(fontsize=12)

    plt.tight_layout()

    loss_path = (
        config.RESULTS_PATH /
        "loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nPlots Saved Successfully")

    print(accuracy_path)

    print(loss_path)


if __name__ == "__main__":

    generate_plots()