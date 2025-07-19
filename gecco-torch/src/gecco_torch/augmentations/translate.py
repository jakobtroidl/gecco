import torch

class RandomTranslation(torch.nn.Module):
    def __init__(self):
        super(RandomTranslation, self).__init__()

    def forward(self, x):
        return x