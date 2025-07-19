import torch
import torch.nn as nn


class RandomJitter(nn.Module):
    def __init__(self, sigma=2.0, clip=2.0):
        """
        Initialize the RandomJitter3D module.
        """
        super(RandomJitter, self).__init__()
        self.sigma = sigma
        self.clip = clip

    def forward(self, point_cloud):
        """
        Apply random jitter to the point cloud.
        """
        noise = torch.clamp(
            self.sigma * torch.randn_like(point_cloud),
            min=-self.clip,
            max=self.clip
        )
        return point_cloud + noise
