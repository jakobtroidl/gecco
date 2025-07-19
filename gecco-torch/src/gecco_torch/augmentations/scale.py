import torch

class GlobalScale(torch.nn.Module):
    def __init__(self, scale: float = 0.00001):
        super(GlobalScale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale