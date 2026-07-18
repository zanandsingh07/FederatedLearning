"""
=========================================================
train.py
Federated Learning Training
TensorFlow 2.21
=========================================================
"""

import os
import random

from src import config

from src.dataset import (
    load_dataset,
    dataset_summary,
    split_dataset,
    verify_no_data_leakage,
    create_tf_dataset,
    create_iid_clients,
)

from src.client import Client
from src.server import Server


def train_federated():

    # --------------------------------------------
    # Load Dataset
    # --------------------------------------------
    print("=" * 70)
    print("Loading Dataset...")
    print("=" * 70)

    df, encoder = load_dataset()

    dataset_summary(df)

    train_df, valid_df, test_df = split_dataset(df)

    verify_no_data_leakage(
        train_df,
        valid_df,
        test_df,
    )

    print()

    print("Train Images      :", len(train_df))
    print("Validation Images :", len(valid_df))
    print("Test Images       :", len(test_df))

    # --------------------------------------------
    # Test Dataset
    # --------------------------------------------

    test_dataset = create_tf_dataset(
        test_df,
        shuffle=False,
    )

    # --------------------------------------------
    # Create Clients
    # --------------------------------------------

    client_dfs = create_iid_clients(train_df)

    clients = []

    for i, client_df in enumerate(client_dfs):

        clients.append(

            Client(

                client_id=i + 1,

                train_df=client_df,

                valid_df=None,

            )

        )

    print()

    print("Total Clients :", len(clients))

    # --------------------------------------------
    # Server
    # --------------------------------------------

    server = Server()

    print()

    print("Server Initialized")

    # --------------------------------------------
    # History
    # --------------------------------------------

    history = {

        "round": [],

        "accuracy": [],

        "loss": []

    }

    best_accuracy = 0

    # --------------------------------------------
    # Global Training
    # --------------------------------------------

    for rnd in range(config.GLOBAL_ROUNDS):

        print("\n")

        print("=" * 70)

        print(f"GLOBAL ROUND {rnd + 1}")

        print("=" * 70)

        # ----------------------------------------

        global_weights = server.get_weights()

        selected_clients = random.sample(

            clients,

            config.CLIENTS_PER_ROUND,

        )

        print()

        print("Selected Clients")

        for c in selected_clients:

            print("Client", c.client_id)

        client_weights = []

        client_sizes = []

        # ----------------------------------------
        # Local Training
        # ----------------------------------------

        for client in selected_clients:

            client.set_weights(global_weights)

            weights, size, _ = client.train()

            client_weights.append(weights)

            client_sizes.append(size)

        # ----------------------------------------
        # FedAvg
        # ----------------------------------------

        server.aggregate(

            client_weights,

            client_sizes,

        )

        # ----------------------------------------
        # Evaluate
        # ----------------------------------------

        metrics = server.evaluate(

            test_dataset

        )

        acc = metrics["accuracy"]

        loss = metrics["loss"]

        print()

        print(f"Global Accuracy : {acc*100:.2f}%")

        print(f"Global Loss     : {loss:.4f}")

        history["round"].append(

            rnd + 1

        )

        history["accuracy"].append(

            acc

        )

        history["loss"].append(

            loss

        )

        # ----------------------------------------
        # Save Best Model
        # ----------------------------------------

        if acc > best_accuracy:

            best_accuracy = acc

            os.makedirs(

                config.MODELS_PATH,

                exist_ok=True,

            )

            server.save_model(

                os.path.join(

                    config.MODELS_PATH,

                    "best_global_model.keras",

                )

            )

            print()

            print("Best Model Saved")

    print()

    print("=" * 70)

    print("Federated Training Completed")

    print("=" * 70)

    return history