import torch
import torch.nn as nn


class SamBinarySegmentationHead(nn.Module):
    """Simple binary segmentation head on top of a frozen SAM encoder.

    Expects encoder features of shape (B, C, H, W) and outputs a 1-channel
    logits mask of shape (B, 1, H, W).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SamBinarySegmenter(nn.Module):
    """Wrapper combining a frozen SAM image encoder with a trainable head.

    The encoder should be an object exposing an `image_encoder` module that
    maps images to feature maps.
    """

    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.sam = sam_model

        # Freeze all SAM params
        for p in self.sam.parameters():
            p.requires_grad = False

        # Infer encoder output channels from a dummy forward later if needed;
        # here we assume the common ViT-B encoder embedding dim of 256.
        # You can adjust this if using a different SAM variant.
        in_channels = 256
        self.head = SamBinarySegmentationHead(in_channels)

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        # image_embeddings: (B, C, H, W) from SAM's image encoder
        return self.head(image_embeddings)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Run SAM image encoder. This does not require gradients."""
        with torch.no_grad():
            return self.sam.image_encoder(images)

    def trainable_parameters(self):
        """Return only parameters of the trainable head for optimization."""
        return self.head.parameters()