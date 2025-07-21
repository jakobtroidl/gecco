import os
import numpy as np
import torch
import lightning.pytorch as pl

from gecco_torch.structs import Example, Neuron
from gecco_torch.data.samplers import ConcatenatedSampler, FixedSampler
from gecco_torch.augmentations.transform import TransformWithInverse

# class TorchNeuronNet:
#     def __init__(
#         self,
#         root: str,
#         split: str,
#         n_points: int = 2048,
#         batch_size: int = 48,
#     ):
#         self.path = os.path.join(root, split)
#         if not os.path.exists(self.path):
#             raise ValueError(f"Path {self.path} does not exist")

#         self.pt = [f for f in os.listdir(self.path) if f.endswith(".pt")]
#         self.n_points = n_points
#         self.batch_size = batch_size

#     def __len__(self):
#         return len(self.pt)

#     def __getitem__(self, index):
#         points = torch.load(os.path.join(self.path, self.pt[index]))
#         points = points.to(torch.float32)
#         perm = torch.randperm(points.shape[0])[: self.n_points]
#         selected = points[perm].clone()

#         # add context as an empty list
#         return Example(selected, [])

class TorchNeuronNet:

    def __init__(
        self,
        root: str,
        split: str,
        n_points: int = 5000,
        batch_size: int = 48,
        voxel_size: list = [4, 4, 40]  # Voxel size for scaling,
    ):
        self.path = os.path.join(root, split)
        if not os.path.exists(self.path):
            raise ValueError(f"Path {self.path} does not exist")

        self.pt = [f for f in os.listdir(self.path) if f.endswith(".pt")]
        self.n_points = n_points
        self.batch_size = batch_size
        print(voxel_size)
        self.transform = TransformWithInverse(voxel_size=voxel_size)

    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, idx):
        file = os.listdir(self.path)[idx]
        data = torch.load(os.path.join(self.path, file))

        points = data["pc"]
        points, T, T_i = self.transform(points)
        # shift = points.mean(dim=0).reshape(1, 3)
        # scale = points.flatten().std().reshape(1, 1)
        # points = (points - shift) / scale

        idx = torch.randperm(points.shape[0])[:self.n_points]
        selected = points[idx].clone()

        mask = data["labels"] == 0
        data["mask"] = mask.squeeze() # add mask for attention

        partial = points[mask.squeeze()]

        if partial.shape[0] > 0:
            idx = torch.randint(0, partial.shape[0], (self.n_points,))
            partial = partial[idx].clone()
        else:
            partial = selected.clone()  # fallback if no partial points

        return Neuron(selected, partial, T, T_i)


class NeuronDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        n_points: int = 2048,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
    ):
        super().__init__()

        self.root = root
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size

    def setup(self, stage=None):
        self.train = TorchNeuronNet(self.root, "train", self.n_points)
        self.val = TorchNeuronNet(self.root, "val", self.n_points)
        self.test = TorchNeuronNet(self.root, "test", self.n_points)

    def train_dataloader(self):
        if self.epoch_size is None:
            kw = dict(
                shuffle=True,
                sampler=None,
            )
        else:
            kw = dict(
                shuffle=False,
                sampler=ConcatenatedSampler(
                    self.train, self.epoch_size * self.batch_size, seed=None
                ),
            )

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kw,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=FixedSampler(self.val, length=None, seed=42),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
