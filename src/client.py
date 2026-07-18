"""
=========================================================
client.py
Federated Learning Client
TensorFlow 2.21
Python 3.13
=========================================================
"""

from src.model import get_model
from src.dataset import create_tf_dataset
from src import config


class Client:
    """
    Federated Learning Client
    """

    def __init__(self, client_id, train_df, valid_df=None):

        self.client_id = client_id

        self.train_df = train_df

        self.valid_df = valid_df

        self.model = get_model()

        self.train_dataset = create_tf_dataset(
            train_df,
            shuffle=True,
        )

        if valid_df is not None:

            self.valid_dataset = create_tf_dataset(
                valid_df,
                shuffle=False,
            )

        else:

            self.valid_dataset = None

        self.num_examples = len(train_df)

    # --------------------------------------------------
    # Set Global Model Weights
    # --------------------------------------------------

    def set_weights(self, weights):

        self.model.set_weights(weights)

    # --------------------------------------------------
    # Get Local Model Weights
    # --------------------------------------------------

    def get_weights(self):

        return self.model.get_weights()

    # --------------------------------------------------
    # Local Training
    # --------------------------------------------------

    def train(self, epochs=None):

        if epochs is None:
            epochs = config.LOCAL_EPOCHS

        print(f"\nTraining Client {self.client_id}")

        history = self.model.fit(

        self.train_dataset,

        epochs=epochs,

        verbose=1,

    )

        return (

        self.get_weights(),

        self.num_examples,

        history.history,

    )
    # --------------------------------------------------
    # Local Evaluation
    # --------------------------------------------------

    def evaluate(self, test_dataset):

        loss, accuracy = self.model.evaluate(

            test_dataset,

            verbose=0,

        )

        return {

            "loss": float(loss),

            "accuracy": float(accuracy),

        }

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self, dataset):

        return self.model.predict(dataset)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def summary(self):

        print()

        print("Client ID :", self.client_id)

        print("Images    :", self.num_examples)

        print()