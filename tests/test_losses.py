import torch

from src.losses import NTXentLoss


device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = NTXentLoss(
    device=device,
    batch_size=8,
    temperature=0.2
)

z1 = torch.randn(8, 32).to(device)
z2 = torch.randn(8, 32).to(device)

loss = loss_fn(z1, z2)

print(loss)