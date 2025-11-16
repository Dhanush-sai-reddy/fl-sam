from typing import List, Tuple

import flwr as fl

from config import TrainConfig
from fl.client import client_fn


def get_strategy(cfg: TrainConfig) -> fl.server.strategy.FedAvg:
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        min_evaluate_clients=0,
    )


def start_server_and_simulation(cfg: TrainConfig):
    strategy = get_strategy(cfg)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

