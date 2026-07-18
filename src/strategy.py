"""
=========================================================
strategy.py
Flower FedAvg Strategy
=========================================================
"""

from flwr.server.strategy import FedAvg


def get_strategy():

    strategy = FedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=3,

        min_evaluate_clients=3,

        min_available_clients=3,

    )

    return strategy