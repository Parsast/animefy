import torch
import torch.nn.functional as F

def rgb_to_xyz(input):
    """
    Convert an RGB image to CIE XYZ.
    Args:
        input: A 4-D Tensor `[N, C, H, W]`.
    Returns:
        A 4-D Tensor `[N, C, H, W]`.
    """
    assert input.dtype in (torch.float16, torch.float32, torch.float64)
    
    kernel = torch.tensor(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        dtype=input.dtype, device=input.device
    )
    
    mask = input > 0.04045
    value = torch.where(mask, torch.pow((input + 0.055) / 1.055, 2.4), input / 12.92)
    
    # The 'matmul' function in PyTorch performs batch matrix multiplication
    # when given tensors with more than two dimensions.
    value = value.permute(0, 2, 3, 1)  # Change to `[N, H, W, C]` to match the kernel shape
    xyz = torch.matmul(value, kernel.t())
    return xyz.permute(0, 3, 1, 2)  # Change back to `[N, C, H, W]`

def rgb_to_lab(input, illuminant="D65", observer="2"):
    """
    Convert an RGB image to CIE LAB.
    Args:
        input: A 4-D Tensor `[N, C, H, W]`.
    Returns:
        A 4-D Tensor `[N, C, H, W]`.
    """
    assert input.dtype in (torch.float16, torch.float32, torch.float64)
    
    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        # ... other illuminants ...
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        # ... other illuminants ...
    }
    coords = torch.tensor(illuminants[illuminant][observer], dtype=input.dtype, device=input.device)
    
    xyz = rgb_to_xyz(input)
    
    xyz = xyz / coords.view(1, 3, 1, 1)  # Reshape for broadcasting
    
    threshold = 0.008856
    linear_mask = xyz > threshold
    scaled_xyz = torch.where(
        linear_mask,
        torch.pow(xyz, 1/3),
        xyz * 7.787 + 16.0 / 116.0
    )
    
    x, y, z = scaled_xyz.unbind(1)
    
    # Vector scaling
    l = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    
    lab = torch.stack([l, a, b], dim=1)
    return lab
