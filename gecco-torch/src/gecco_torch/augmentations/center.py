import torch

class Center(torch.nn.Module):
    def __init__(self):
        super(Center, self).__init__()

    def forward(self, x):
        return x - x.mean(dim=0)
    

class CenterTransform(torch.nn.Module):
    def __init__(self):
        super(CenterTransform, self).__init__()

    def forward(self, x):
        centroid = x.mean(axis=0)
        T = torch.eye(4)
        T[:3, 3] = -centroid
        return T