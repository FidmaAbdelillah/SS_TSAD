import numpy as np

from src.anomaly import (
    fit_mahalanobis,
    mahalanobis_scores,
    contiguous_block_scores,
)


# Healthy embeddings
Z_train = np.random.randn(
    100,
    128
)

mu, L = fit_mahalanobis(
    Z_train
)

scores = mahalanobis_scores(
    Z_train,
    mu,
    L
)

print(scores.shape)


block_scores = contiguous_block_scores(
    scores,
    block_length=5,
    reduction="mean"
)

print(block_scores.shape)