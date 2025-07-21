import torch

from gecco_torch.augmentations.scale import GlobalScaleTransform
from gecco_torch.augmentations.center import CenterTransform
from gecco_torch.utils import apply_transform
    
class TransformWithInverse(torch.nn.Module):
    def __init__(self, voxel_size: list = [1.0, 1.0, 1.0]):
        super(TransformWithInverse, self).__init__()
        self.transforms = [
            GlobalScaleTransform(voxel_size[0], voxel_size[1], voxel_size[2]),
            CenterTransform(),
            GlobalScaleTransform(),
        ]

    def forward(self, x): 
        """
        Apply the transformations to the input tensor x.
        If inverse is True, apply the inverse transformations.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        T = torch.eye(4, dtype=x.dtype, device=x.device)
        for transform in self.transforms:
            t = transform(x)
            T = t @ T
        x = apply_transform(T, x)
        T_i = torch.inverse(T)
        return x, T, T_i