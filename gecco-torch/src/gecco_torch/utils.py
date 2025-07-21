import torch

def apply_transform(T, points):
    """
    Apply the transformation to the points.
    Args:
        points (torch.Tensor): The input points of shape (N, 3).
        T (torch.Tensor): The transformation matrix of shape (4, 4).
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # make points homogeneous
    points = torch.cat([points, torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)], dim=-1)
    T = T.to(points.dtype).to(points.device)  # ensure T is on the same device and dtype as points
    
    # apply transformation
    points = (T @ points.T).T
    
    return points[:, :3]