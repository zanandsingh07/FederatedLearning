"""
=========================================================
evaluate.py
Evaluate Best Global Model
=========================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src import config
from src.dataset import (
    load_dataset,
    split_dataset,
    create_tf_dataset,
)

def evaluate_global_model():

    print("=" * 60)
    print("Loading Best Global Model")
    print("=" * 60)

    # ---------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------

    df, encoder = load_dataset()

    train_df, valid_df, test_df = split_dataset(df)

    test_dataset = create_tf_dataset(
        test_df,
        shuffle=False,
    )

    # ---------------------------------------------------
    # Load Model
    # ---------------------------------------------------

    model_path = config.MODELS_PATH / "best_global_model.keras"

    model = tf.keras.models.load_model(model_path)

    print()

    print("Model Loaded Successfully")

    print(model_path)

    # ---------------------------------------------------
    # Evaluate
    # ---------------------------------------------------

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )
    # ---------------------------------------------------
# Predictions
# ---------------------------------------------------

    print("\nGenerating Predictions...")

    y_true = test_df["label"].values

    predictions = model.predict(
    test_dataset,
    verbose=1,
  )

    y_pred = np.argmax(predictions, axis=1)

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

    cm = confusion_matrix(
    y_true,
    y_pred,
  )

    labels = encoder.classes_

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False,
    )

    plt.title("Confusion Matrix")

    os.makedirs(
    config.RESULTS_PATH,
    exist_ok=True,
    )

    save_path = config.RESULTS_PATH / "confusion_matrix.png"

    plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    )

    plt.show()

    print("\nConfusion Matrix Saved")

    print(save_path)

    print()

    print("=" * 60)

    print("TEST RESULTS")

    print("=" * 60)

    print(f"Test Loss     : {loss:.4f}")

    print(f"Test Accuracy : {accuracy*100:.2f}%")

    return model, test_dataset, encoder


if __name__ == "__main__":

    evaluate_global_model()