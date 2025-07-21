import torch

def apply_transform(T, points):
    """
    Apply the transformation to the points.
    Args:
        points (torch.Tensor): The input points of shape (B, N, 3) or (N, 3).
        T (torch.Tensor): The transformation matrix of shape (B, 4, 4) or (4, 4).
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Handle different input shapes
    original_shape = points.shape
    squeeze_batch = False
    
    if points.dim() == 2:
        # Single batch: (N, 3) -> (1, N, 3)
        points = points.unsqueeze(0)
        squeeze_batch = True
    elif points.dim() == 3:
        # Already batched: (B, N, 3)
        pass
    else:
        raise ValueError("Input points must be of shape (B, N, 3) or (N, 3)")
    
    if points.shape[-1] != 3:
        raise ValueError("Last dimension of points must be 3")
    
    # Handle transformation matrix shapes
    if T.dim() == 2:
        # Single transformation: (4, 4) -> (1, 4, 4) or broadcast to batch size
        if squeeze_batch:
            T = T.unsqueeze(0)
        else:
            T = T.unsqueeze(0).expand(points.shape[0], -1, -1)
    elif T.dim() == 3:
        # Already batched: (B, 4, 4)
        if T.shape[0] != points.shape[0]:
            raise ValueError(f"Batch size mismatch: points {points.shape[0]}, T {T.shape[0]}")
    else:
        raise ValueError("Transformation matrix T must be of shape (4, 4) or (B, 4, 4)")
    
    if T.shape[-2:] != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")

    # Make points homogeneous: (B, N, 3) -> (B, N, 4)
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    points_homo = torch.cat([points, ones], dim=-1)
    
    # Ensure T is on the same device and dtype as points
    T = T.to(points.dtype).to(points.device)
    
    # Apply transformation using batch matrix multiplication
    # points_homo: (B, N, 4), T: (B, 4, 4)
    # We want: T @ points_homo^T for each batch, then transpose back
    # Result: (B, N, 4)
    transformed = torch.bmm(points_homo, T.transpose(-2, -1))
    
    # Extract xyz coordinates and handle output shape
    result = transformed[..., :3]
    
    if squeeze_batch:
        result = result.squeeze(0)
    
    return result