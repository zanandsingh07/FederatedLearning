"""
=========================================================
server.py
Federated Learning Server
TensorFlow 2.21
=========================================================
"""

import numpy as np

from src.model import get_model


class Server:
    """
    Federated Learning Server
    """

    def __init__(self):

        self.global_model = get_model()

    # -------------------------------------------------
    # Get Global Model Weights
    # -------------------------------------------------

    def get_weights(self):

        return self.global_model.get_weights()

    # -------------------------------------------------
    # Update Global Model Weights
    # -------------------------------------------------

    def set_weights(self, weights):

        self.global_model.set_weights(weights)

    # -------------------------------------------------
    # FedAvg Aggregation
    # -------------------------------------------------

    def aggregate(self, client_weights, client_sizes):
        """
        Weighted FedAvg Aggregation

        Parameters
        ----------
        client_weights : List of model weights

        client_sizes : Number of samples on each client
        """

        total_examples = sum(client_sizes)

        new_weights = []

        for layer in zip(*client_weights):

            weighted_layer = np.zeros_like(layer[0])

            for weights, size in zip(layer, client_sizes):

                weighted_layer += weights * (
                    size / total_examples
                )

            new_weights.append(weighted_layer)

        self.set_weights(new_weights)

    # -------------------------------------------------
    # Evaluate Global Model
    # -------------------------------------------------

    def evaluate(self, test_dataset):

        loss, accuracy = self.global_model.evaluate(

            test_dataset,

            verbose=0,

        )

        return {

            "loss": float(loss),

            "accuracy": float(accuracy),

        }

    # -------------------------------------------------
    # Save Global Model
    # -------------------------------------------------

    def save_model(self, filepath):

        self.global_model.save(filepath)

        print()

        print("=" * 60)

        print("Global Model Saved")

        print(filepath)

        print("=" * 60)