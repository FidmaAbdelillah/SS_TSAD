import torch

from src.encoder import BaseEncoderCNN


model = BaseEncoderCNN(
    input_channels=1,
    final_out_channels=128
)

x = torch.randn(4, 2048)

y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)