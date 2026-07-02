import torch

from src.augmentations import (
    weak_view,
    strong_view,
    time_mask
)


x = torch.randn(8, 2048)

x_weak = weak_view(x)
x_strong = strong_view(x)
x_masked = time_mask(x)

print(x.shape)
print(x_weak.shape)
print(x_strong.shape)
print(x_masked.shape)