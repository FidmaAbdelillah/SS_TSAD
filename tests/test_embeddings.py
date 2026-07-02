import numpy as np

from src.trainer import pretrain_tstcc
from src.embeddings import extract_context_vector


# Dummy dataset
X = np.random.randn(
    128,
    2048
).astype(np.float32)


# Train quickly
model = pretrain_tstcc(
    X,
    epochs=1,
    batch_size=16,
    tc_timesteps=8,
)


# Extract embeddings
Z = extract_context_vector(
    model,
    X,
    batch_size=16,
    t_fixed=20,
)


print(Z.shape)