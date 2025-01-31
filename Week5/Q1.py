import torch
import torch.nn.functional as F

# Task 1: Convolution operation on an image with a random kernel
image = torch.rand(6, 6)  # Image of shape (6, 6)
print("Original image:\n", image)

# Add a new dimension along 0th dimension to get (1, 6, 6) for batch size
image = image.unsqueeze(dim=0)  # Shape becomes (1, 6, 6)
image = image.unsqueeze(dim=0)  # Shape becomes (1, 1, 6, 6)
print("Image shape after unsqueeze:", image.shape)

# Kernel of size (3, 3)
kernel = torch.ones(3, 3)  # Kernel shape is (3, 3)
print("Kernel:\n", kernel)

# Add new dimensions to the kernel for out_channels and in_channels
kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)  # Shape becomes (1, 1, 3, 3)
print("Kernel shape after unsqueeze:", kernel.shape)

# Perform the convolution operation (stride=1, padding=0)
outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("Output image after convolution:\n", outimage)
print("Output image shape:", outimage.shape)
