import torch
    
class RandomFlip(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super(RandomFlip, self).__init__()
        self.p = p

    def forward(self, x):
        # flips N, 3 point cloud with probability p randomly along each axis
        for i in range(x.shape[1]):
            if torch.rand(1) < self.p:
                x[:, i] = -x[:, i]

        return x