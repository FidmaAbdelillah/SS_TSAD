import torch

from src.tc import TC


device = "cuda" if torch.cuda.is_available() else "cpu"


model = TC(
    device=device,
    final_out_channels=128,
    timesteps=16,
    hidden_dim=256,
).to(device)


x1 = torch.randn(
    32,
    128,
    50
).to(device)

x2 = torch.randn(
    32,
    128,
    50
).to(device)


nce_loss, projection = model(x1, x2)


print("NCE loss:", nce_loss.item())
print("Projection shape:", projection.shape)