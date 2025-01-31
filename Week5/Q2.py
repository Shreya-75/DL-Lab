import torch
import torch.nn as nn
import torch.nn.functional as F

# Task 2: Apply Conv2d using nn.Conv2d and F.conv2d
image = torch.rand(1, 1, 6, 6)  # Image of shape (1, 1, 6, 6)

# Using nn.Conv2d
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
out_image_conv2d = conv_layer(image)
print("Output with nn.Conv2d:", out_image_conv2d.shape)

# Implementing the same with F.conv2d
kernel = torch.ones(1, 1, 3, 3)  # Kernel of size (1, 1, 3, 3)
out_image_f_conv2d = F.conv2d(image, kernel, stride=1, padding=0)
print("Output with F.conv2d:", out_image_f_conv2d.shape)
