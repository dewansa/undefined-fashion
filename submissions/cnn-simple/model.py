"""Two strided convolutions for Fashion-MNIST. Feature maps go from 28 to 9 to 4."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class PatchConv2d(nn.Conv2d):
    """A valid convolution, evaluated as one matrix multiply over all image patches."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = self.kernel_size
        patches = images.unfold(2, height, self.stride[0]).unfold(3, width, self.stride[1])
        # Each row is a patch. Every location uses the same bank of learned filters.
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        features = F.linear(patches, self.weight.flatten(1), self.bias)
        return features.movedim(-1, 1)


class FashionCNN(nn.Sequential):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        scores = super().forward(images)
        if self.training:
            return scores
        # At prediction time, average the two orientations used during training.
        return (scores + super().forward(images.flip((-1,)))) * 0.5


def train(train_set: Dataset) -> nn.Module:
    if hasattr(train_set, "data") and hasattr(train_set, "targets"):
        images = train_set.data.unsqueeze(1).float().div_(255)
        labels = train_set.targets
    else:
        images, labels = next(iter(DataLoader(train_set, batch_size=len(train_set))))

    # Overlapping 4 x 4 patches (stride 3) see more positions than stride 4, and the
    # narrower layers keep the compute just under the stride-4 version.
    # BatchNorm adds no counted FLOPs and lets the network train with a larger step size.
    model = FashionCNN(
        PatchConv2d(1, 16, kernel_size=4, stride=3),  # 28 x 28 -> 9 x 9
        nn.BatchNorm2d(16),
        nn.ReLU(),
        PatchConv2d(16, 48, kernel_size=3, stride=2),  # 9 x 9 -> 4 x 4
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(48 * 4 * 4, 48),
        nn.BatchNorm1d(48),
        nn.ReLU(),
        nn.Linear(48, 10),
    )
    epochs, batch_size = 28, 512
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012)
    # Warm up the step size, then lower it smoothly to almost zero by the end.
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.012, epochs=epochs, steps_per_epoch=-(-len(images) // batch_size)
    )
    for _ in range(epochs):
        order = torch.randperm(len(images))
        for step, indices in enumerate(order.split(batch_size)):
            batch = images[indices]
            if step % 2:
                batch = batch.flip((-1,))
            if step % 3 == 0:
                top, left = torch.randint(3, (2,)).tolist()
                batch = F.pad(batch, (1, 1, 1, 1))[:, :, top : top + 28, left : left + 28]

            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(batch), labels[indices]).backward()
            optimizer.step()
            schedule.step()
    return model.eval()
