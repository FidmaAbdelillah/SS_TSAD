import numpy as np

from src.trainer import pretrain_tstcc


X = np.random.randn(
    128,
    2048
).astype(np.float32)


model = pretrain_tstcc(
    X,
    epochs=2,
    batch_size=16,
    tc_timesteps=8,
    verbose=True,
)