"""
simulation.py
Coordinates dataset preparation and client creation.
"""

from src.dataset import (
    load_dataset,
    split_dataset,
    verify_no_data_leakage,
    create_iid_clients,
)
from src.server import get_server_config, get_strategy


def prepare_simulation():
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df, encoder = load_dataset()

    print(f"Total Images : {len(df)}")

    train_df, valid_df, test_df = split_dataset(df)

    verify_no_data_leakage(
        train_df,
        valid_df,
        test_df,
    )

    client_datasets = create_iid_clients(train_df)

    print()

    print(f"Total Clients : {len(client_datasets)}")

    for idx, client_df in enumerate(client_datasets, start=1):
        print(
            f"Client {idx:02d} : "
            f"{len(client_df)} images"
        )

    strategy = get_strategy()

    server_config = get_server_config()

    return (
        client_datasets,
        test_df,
        strategy,
        server_config,
    )