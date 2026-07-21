"""
=========================================================
evaluate.py
Evaluate Best Global Model
=========================================================
"""

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

    print()

    print("=" * 60)

    print("TEST RESULTS")

    print("=" * 60)

    print(f"Test Loss     : {loss:.4f}")

    print(f"Test Accuracy : {accuracy*100:.2f}%")

    return model, test_dataset, encoder


if __name__ == "__main__":

    evaluate_global_model()