import torch 

class GlobalScaleTransform(torch.nn.Module):
    def __init__(self, scale_x: float = 0.000005, scale_y: float = 0.000005, scale_z: float = 0.000005):
        super(GlobalScaleTransform, self).__init__()
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

    def forward(self, x):
        T = torch.eye(4, dtype=x.dtype, device=x.device)
        T[0, 0] = self.scale_x
        T[1, 1] = self.scale_y
        T[2, 2] = self.scale_z

        return T