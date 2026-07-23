"""
=========================================================
train.py
Federated Learning Training
TensorFlow 2.21
=========================================================
"""

import os
import random
import pandas as pd

from src import config

from src.dataset import (
    load_dataset,
    dataset_summary,
    split_dataset,
    verify_no_data_leakage,
    create_tf_dataset,
    create_iid_clients,
    create_non_iid_clients,
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
    # Validation Dataset
    # --------------------------------------------

    validation_dataset = create_tf_dataset(
    valid_df,
    shuffle=False,
)

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

    if config.DATA_DISTRIBUTION == "iid":

      client_dfs = create_iid_clients(train_df)

    else:

     client_dfs = create_non_iid_clients(train_df)
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
        print("\n" + "=" * 70)
        print("CLIENT DATASET SUMMARY")
        print("=" * 70)

    for i, client_df in enumerate(client_dfs):

        print(
            f"Client {i+1:02d} : "
            f"{len(client_df)} images"
        )
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

    "train_accuracy": [],

    "train_loss": [],

    "validation_accuracy": [],

    "validation_loss": []

}

    best_accuracy = 0

    # --------------------------------------------
    # Global Training
    # --------------------------------------------
    best_accuracy = 0.0

    wait = 0
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
        train_acc_list = []
        train_loss_list = []
        # ----------------------------------------
        # Local Training
        # ----------------------------------------

        for client in selected_clients:

            print(f"\nTraining Client {client.client_id}")

            client.set_weights(global_weights)

            weights, size, history_local = client.train()

            client_weights.append(weights)

            client_sizes.append(size)

        # Last epoch metrics
            train_acc_list.append(
            history_local["accuracy"][-1]
             )

            train_loss_list.append(
            history_local["loss"][-1]
            )
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
         validation_dataset
        )

        acc = metrics["accuracy"]

        loss = metrics["loss"]
        avg_train_acc = sum(train_acc_list) / len(train_acc_list)

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)

        print()

        print("=" * 50)
        print("ROUND SUMMARY")
        print("=" * 50)

        print(f"Average Train Accuracy : {avg_train_acc*100:.2f}%")
        print(f"Average Train Loss     : {avg_train_loss:.4f}")

        print()

        print(f"Validation Accuracy    : {acc*100:.2f}%")
        print(f"Validation Loss        : {loss:.4f}")
        
        history["round"].append(rnd + 1)

        history["train_accuracy"].append(avg_train_acc)

        history["train_loss"].append(avg_train_loss)

        history["validation_accuracy"].append(acc)

        history["validation_loss"].append(loss)

        # ----------------------------------------
        # Save Best Model
        # ----------------------------------------

        if acc > best_accuracy + config.MIN_DELTA:

            best_accuracy = acc

            wait = 0

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
            print("Best Global Model Saved")

        else:

            wait += 1

            print()

            print(
                f"No improvement "
                f"({wait}/{config.GLOBAL_PATIENCE})"
            )
        if (
            config.GLOBAL_EARLY_STOPPING
            and wait >= config.GLOBAL_PATIENCE
        ):

            print()

            print("=" * 60)
            print("GLOBAL EARLY STOPPING TRIGGERED")
            print("=" * 60)

            print(
                f"Training stopped after "
                f"{rnd + 1} communication rounds."
            )

            break

    print()

    print("=" * 70)

    print("Federated Training Completed")

# ===============================================
# Save Training History
# ===============================================

   # ===============================================
# Save Training History
# ===============================================

    os.makedirs(
    config.RESULTS_PATH,
    exist_ok=True,
    )

    history_df = pd.DataFrame(history)

    history_path = config.RESULTS_PATH / "history.csv"

    history_df.to_csv(
    history_path,
    index=False,
)
    print("\nTraining history saved.")
    print(history_path)
    print("=" * 70)

    return history
if __name__ == "__main__":
    train_federated()

    print("Step 1")
    df, encoder = load_dataset()

    print("Step 2")
    dataset_summary(df)

    print("Step 3")
    train_df, valid_df, test_df = split_dataset(df)

    print("Step 4")
    client_dfs = create_non_iid_clients(train_df)

    print("Step 5")