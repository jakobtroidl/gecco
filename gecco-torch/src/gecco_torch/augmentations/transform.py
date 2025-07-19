import torch

from gecco_torch.augmentations.rotate import RandomRotation
from gecco_torch.augmentations.center import Center
from gecco_torch.augmentations.flip import RandomFlip
from gecco_torch.augmentations.jitter import RandomJitter
from gecco_torch.augmentations.scale import GlobalScale
from gecco_torch.augmentations.shear import RandomShear

class Transform(torch.nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.transforms = torch.nn.Sequential(
            Center(),
            # RandomJitter(),
            GlobalScale(),
            # RandomFlip(),
            # RandomRotation(allow_axes=(True, True, True), max_rotation_angle=math.pi / 4),
            # RandomShear(max_shear=0.1),
        )

    def forward(self, x):
        return self.transforms(x)