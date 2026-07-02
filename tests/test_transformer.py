import torch

from src.transformer import SeqTransformer


model = SeqTransformer(
    patch_size=128,
    dim=256,
    depth=4,
    heads=8,
    mlp_dim=512
)

x = torch.randn(
    16,   # batch
    50,   # timesteps
    128   # features
)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)