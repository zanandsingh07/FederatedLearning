"""
=========================================================
task.py
Shared ML functions for Flower Clients
Flower 1.32.1
=========================================================
"""

from src.model import get_model
from src.dataset import create_tf_dataset


def load_model():
    """
    Create CNN model.
    """
    return get_model()


def train(model, train_df, epochs):

    train_dataset = create_tf_dataset(
        train_df,
        shuffle=True,
    )

    history = model.fit(

        train_dataset,

        epochs=epochs,

        verbose=1,

    )

    return history


def evaluate(model, test_df):

    test_dataset = create_tf_dataset(

        test_df,

        shuffle=False,

    )

    loss, accuracy = model.evaluate(

        test_dataset,

        verbose=0,

    )

    return loss, accuracy