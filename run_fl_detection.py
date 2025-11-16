import argparse

from config import TrainConfig
from fl.server import start_server_and_simulation


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="FL + Segment Anything (SAM) binary segmentation on COCO with Flower")
    parser.add_argument("--data-root", type=str, default="./data/coco2017")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    start_server_and_simulation(cfg)

