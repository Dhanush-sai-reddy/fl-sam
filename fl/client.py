from typing import Dict, List

import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import flwr as fl
from segment_anything import sam_model_registry

# Ensure project root is on sys.path so that `datasets` and `models` can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TrainConfig
from datasets.coco_dataset import get_coco_binary_segmentation
from models.sam_segmentation import SamBinarySegmenter


def get_client_dataloader(cfg: TrainConfig, cid: int):
    dataset = get_coco_binary_segmentation(cfg.data_root, split="train")
    n = len(dataset)
    shard_size = n // cfg.num_clients
    start = cid * shard_size
    end = n if cid == cfg.num_clients - 1 else (cid + 1) * shard_size
    subset = Subset(dataset, list(range(start, end)))
    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    return loader


def train_one_epoch(model: SamBinarySegmenter, optimizer: torch.optim.Optimizer, data_loader, device: torch.device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            image_embeddings = model.encode_image(images)
        logits = model(image_embeddings)

        # Resize masks if needed to match logits spatial size
        if masks.shape[2:] != logits.shape[2:]:
            masks_resized = torch.nn.functional.interpolate(masks, size=logits.shape[2:], mode="nearest")
        else:
            masks_resized = masks

        loss = criterion(logits, masks_resized)
        loss.backward()
        optimizer.step()


class COCOSamSegClient(fl.client.NumPyClient):
    def __init__(self, cid: str, cfg: TrainConfig, sam_checkpoint: str = "sam_vit_b.pth"):
        self.cid = int(cid)
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(self.device)
        self.model = SamBinarySegmenter(sam).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=cfg.lr)
        self.train_loader = get_client_dataloader(cfg, self.cid)

    def get_parameters(self, config: Dict[str, str]):  # type: ignore[override]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List):  # type: ignore[override]
        state_dict = self.model.state_dict()
        for (k, v), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):  # type: ignore[override]
        self.set_parameters(parameters)
        for _ in range(self.cfg.local_epochs):
            train_one_epoch(self.model, self.optimizer, self.train_loader, self.device)
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):  # type: ignore[override]
        # For now, we skip heavy evaluation and return dummy metrics.
        self.set_parameters(parameters)
        loss = 0.0
        return float(loss), len(self.train_loader.dataset), {"loss": float(loss)}


def client_fn(cid: str):
    cfg = TrainConfig()
    return COCOSamSegClient(cid, cfg)

