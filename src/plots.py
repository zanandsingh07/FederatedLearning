"""
plots.py
Generate Training Curves
"""

import matplotlib.pyplot as plt
import pandas as pd

from src import config

history = pd.read_csv(
    config.RESULTS_PATH / "history.csv"
)

# ------------------------------------------
# Accuracy
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history["round"],
    history["accuracy"],
    marker="o",
    linewidth=2,
)

plt.title("Global Accuracy vs Communication Rounds")

plt.xlabel("Communication Round")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(
    config.RESULTS_PATH /
    "accuracy_rounds.png",
    dpi=300,
)

plt.show()

# ------------------------------------------
# Loss
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history["round"],
    history["loss"],
    marker="o",
    linewidth=2,
)

plt.title("Global Loss vs Communication Rounds")

plt.xlabel("Communication Round")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig(
    config.RESULTS_PATH /
    "loss_rounds.png",
    dpi=300,
)

plt.show()

print("Graphs saved successfully.")