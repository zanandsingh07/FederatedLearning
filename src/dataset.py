"""
dataset.py
Handles dataset loading, preprocessing, splitting, and client creation.
"""

import os
import random
from pathlib import Path
from src import config
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

def load_dataset():
    """
    Load all images and labels into a pandas DataFrame.

    Returns
    -------
    df : pandas.DataFrame
    encoder : LabelEncoder
    """

    filepaths = []
    labels = []

    dataset_path = Path(config.DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    for class_folder in sorted(dataset_path.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        for image_file in class_folder.iterdir():

            if image_file.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            ]:
                continue

            filepaths.append(str(image_file))
            labels.append(class_name)

    df = pd.DataFrame({
        "filepath": filepaths,
        "class_name": labels
    })

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(df["class_name"])

    return df, encoder

def dataset_summary(df):
    """
    Display basic dataset statistics.
    """

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total Images : {len(df)}")
    print(f"Total Classes: {df['class_name'].nunique()}")

    print("\nClass Distribution:")
    print(df["class_name"].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("=====================================\n")

    from sklearn.model_selection import train_test_split


def split_dataset(df):
    """
    Split dataset into Train, Validation and Test.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, valid_df, test_df

def verify_no_data_leakage(train_df, valid_df, test_df):
    """
    Ensure there are no overlapping images.
    """

    train = set(train_df["filepath"])
    valid = set(valid_df["filepath"])
    test = set(test_df["filepath"])

    print("Train ∩ Validation :", len(train & valid))
    print("Train ∩ Test       :", len(train & test))
    print("Validation ∩ Test  :", len(valid & test))

def process_image(filepath, label):
    """
    Read and preprocess one image.
    """

    image = tf.io.read_file(filepath)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        config.IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label

def create_tf_dataset(df, shuffle=False):
    """
    Convert DataFrame to TensorFlow Dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["filepath"].values,
            df["label"].values,
        )
    )

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(
            len(df),
            seed=config.RANDOM_SEED,
        )

    dataset = dataset.batch(config.BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
def create_iid_clients(train_df):
    """
    Split the training dataset into IID clients.
    """

    train_df = train_df.sample(
        frac=1,
        random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    client_size = len(train_df) // config.NUM_CLIENTS

    client_datasets = []

    start = 0

    for i in range(config.NUM_CLIENTS):

        if i == config.NUM_CLIENTS - 1:
            client_df = train_df.iloc[start:]
        else:
            end = start + client_size
            client_df = train_df.iloc[start:end]
            start = end

        client_df = client_df.reset_index(drop=True)

        client_datasets.append(client_df)

    return client_datasets