import torch

from src.model import TSTCC


device = "cuda" if torch.cuda.is_available() else "cpu"


model = TSTCC(
    device=device,
    batch_size=16,
).to(device)


x1 = torch.randn(
    16,
    2048,
).to(device)

x2 = torch.randn(
    16,
    2048,
).to(device)


loss, logs = model(x1, x2)


print("Loss:", loss.item())

print("\nLogs:")

for key, value in logs.items():
    print(key, value.item())