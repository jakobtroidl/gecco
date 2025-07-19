import torch
import torch.nn as nn
import math


class RandomRotation(nn.Module):
    def __init__(self, max_rotation_angle=2 * math.pi, allow_axes=(True, True, True)):
        """
        Initialize the RandomRotation module.

        Args:
            max_rotation_angle (float): Maximum rotation angle in radians. Default is 2*pi (full rotation).
            allow_axes (tuple): A tuple of three booleans (allow_x, allow_y, allow_z) specifying
                                which axes are allowed to rotate.
        """
        super().__init__()
        self.max_rotation_angle = max_rotation_angle
        self.allow_axes = allow_axes

    def forward(self, point_cloud):
        """
        Apply a random 3D rotation with constrained axes to the input point cloud.

        Args:
            point_cloud (torch.Tensor): Input tensor of shape (N, 3).

        Returns:
            torch.Tensor: Rotated point cloud of shape (N, 3).
        """
        rotation_matrix = self._constrained_rotation_matrix()
        return torch.matmul(point_cloud, rotation_matrix)

    def _constrained_rotation_matrix(self):
        """
        Generate a rotation matrix with random angles constrained to allowed axes.

        Returns:
            torch.Tensor: A 3x3 rotation matrix.
        """
        R = torch.eye(3)

        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            if self.allow_axes[i]:
                angle = torch.tensor(torch.rand(1).item() * self.max_rotation_angle)
                R_axis = self._rotation_matrix_single_axis(axis, angle)
                R = R_axis @ R  # compose rotation matrices
        return R

    def _rotation_matrix_single_axis(self, axis, angle):
        """
        Generate a rotation matrix for a single axis.

        Args:
            axis (str): 'x', 'y', or 'z'.
            angle (float): Rotation angle in radians.

        Returns:
            torch.Tensor: A 3x3 rotation matrix.
        """
        c, s = torch.cos(angle), torch.sin(angle)

        if axis == 'x':
            return torch.tensor([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        elif axis == 'y':
            return torch.tensor([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif axis == 'z':
            return torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'")
