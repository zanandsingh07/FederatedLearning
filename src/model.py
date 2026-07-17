"""
=========================================================
model.py
CNN Model for Kidney Disease Classification
TensorFlow 2.21
=========================================================
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

from src import config


def create_model():
    """
    Create CNN model.
    """

    model = Sequential([

        Input(shape=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            3
        )),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),


        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),


        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),


        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            config.NUM_CLASSES,
            activation="softmax"
        )

    ])

    return model


def compile_model(model):
    """
    Compile CNN model.
    """

    model.compile(

        optimizer=Adam(
            learning_rate=config.LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


def get_model():
    """
    Create + Compile
    """

    model = create_model()

    compile_model(model)

    return model