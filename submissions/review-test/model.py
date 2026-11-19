"""Test submission: nearest centroid. It stores the average image of each class
and labels a new image with the closest average. It has no trained parameters."""

import torch
from torch import nn
from torch.utils.data import Dataset


class NearestCentroid(nn.Module):
    def __init__(self, centroids: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("centroids", centroids)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return -torch.cdist(images.flatten(1), self.centroids)


def train(train_set: Dataset) -> nn.Module:
    images = train_set.data.float().flatten(1) / 255
    labels = train_set.targets
    centroids = torch.stack([images[labels == c].mean(dim=0) for c in range(10)])
    return NearestCentroid(centroids)
