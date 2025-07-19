import torch

class Center(torch.nn.Module):
    def __init__(self):
        super(Center, self).__init__()

    def forward(self, x):
        return x - x.mean(dim=0)