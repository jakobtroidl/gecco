import torch
import torch.nn as nn

class RandomShear(nn.Module):
    def __init__(self, max_shear=0.1):
        """
        Initialize the RandomShear module.

        Args:
            shear_matrix (torch.Tensor or None): A 3x3 tensor defining the shear transformation.
                                                 If None, uses identity (no shear).
        """
        super().__init__()
        self.matrix = self._shear_matrix(max_shear)
        assert self.matrix.shape == (3, 3), "Shear matrix must be of shape (3, 3)"
        self.register_buffer("shear_matrix", self.matrix)


    def _shear_matrix(self, max_shear):
        shear_factors = torch.empty(6).uniform_(-max_shear, max_shear)
        # print(f"Shear factors: {shear_factors}")
        return torch.tensor([
            [1.0, shear_factors[0], shear_factors[1]],
            [shear_factors[2], 1.0, shear_factors[3]],
            [shear_factors[4], shear_factors[5], 1.0]
        ])

    def forward(self, point_cloud):
        """
        Apply the shear transformation to the point cloud.

        Args:
            point_cloud (torch.Tensor): Input tensor of shape (N, 3).

        Returns:
            torch.Tensor: Sheared point cloud of shape (N, 3).
        """
        return torch.matmul(point_cloud, self.matrix.T)

