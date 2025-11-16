import os
from typing import Callable, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision import transforms as T


def _default_transforms(is_train: bool = True) -> Callable:
    t = []
    t.append(T.ToTensor())
    # You can add more augmentation for training here if needed
    return T.Compose(t)


def get_coco_detection(root: str, year: str = "2017", split: str = "train") -> Tuple[CocoDetection, int]:
    """Create a CocoDetection dataset and return it along with num_classes.

    Assumes COCO-style directory structure under root.
    """
    img_dir = os.path.join(root, f"{split}{year}")
    ann_file = os.path.join(root, "annotations", f"instances_{split}{year}.json")

    dataset = CocoDetection(img_dir, ann_file, transforms=_default_transforms(split == "train"))

    # COCO has 80 object categories; +1 for background
    num_classes = 81
    return dataset, num_classes


class CocoBinarySegmentation(CocoDetection):
    """COCO dataset providing image and binary foreground mask.

    Any annotated object pixel is treated as foreground (1), background is 0.
    """

    def __init__(self, root: str, year: str = "2017", split: str = "train", transforms: Callable | None = None):
        img_dir = os.path.join(root, f"{split}{year}")
        ann_file = os.path.join(root, "annotations", f"instances_{split}{year}.json")
        super().__init__(img_dir, ann_file)

        self.coco_api = COCO(ann_file)
        self.transforms = transforms or _default_transforms(split == "train")

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img_id = self.ids[index]
        ann_ids = self.coco_api.getAnnIds(imgIds=img_id)
        anns = self.coco_api.loadAnns(ann_ids)

        height, width = img.size[1], img.size[0]
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            ann_mask = self.coco_api.annToMask(ann)
            mask = np.maximum(mask, ann_mask.astype(np.uint8))

        if self.transforms is not None:
            img = self.transforms(img)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask_tensor


def get_coco_binary_segmentation(root: str, year: str = "2017", split: str = "train") -> CocoBinarySegmentation:
    return CocoBinarySegmentation(root=root, year=year, split=split)

