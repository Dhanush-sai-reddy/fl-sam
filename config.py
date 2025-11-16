from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_root: str = "./data/coco2017"
    num_clients: int = 3
    num_rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 2
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    sam_rho: float = 0.05
    num_workers: int = 2
    device: str = "cuda"  # or "cpu"

