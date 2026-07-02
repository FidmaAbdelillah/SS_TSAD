# src/anomaly.py

import numpy as np


# ==========================================================
# Mahalanobis Distance
# ==========================================================

def fit_mahalanobis(
    Z_healthy: np.ndarray,
    eps: float = 1e-3,
):
    """
    Fit a Mahalanobis model on healthy embeddings.

    Parameters
    ----------
    Z_healthy : np.ndarray
        Healthy embeddings of shape (N, D)

    eps : float
        Regularization term.

    Returns
    -------
    mu : np.ndarray
        Mean vector.

    L : np.ndarray
        Cholesky decomposition of covariance matrix.
    """

    Z = Z_healthy.astype(np.float64)

    mu = Z.mean(axis=0)

    X_centered = Z - mu

    covariance = (
        X_centered.T @ X_centered
    ) / max(len(Z) - 1, 1)

    covariance += eps * np.eye(covariance.shape[0])

    L = np.linalg.cholesky(covariance)

    return mu, L


def mahalanobis_scores(
    Z: np.ndarray,
    mu: np.ndarray,
    L: np.ndarray,
):
    """
    Compute Mahalanobis distances.
    """

    Z = Z.astype(np.float64)

    X_centered = Z - mu

    whitened = np.linalg.solve(
        L,
        X_centered.T
    )

    scores = np.sum(
        whitened ** 2,
        axis=0
    )

    return scores


# ==========================================================
# Thresholding
# ==========================================================

def compute_threshold(
    healthy_scores: np.ndarray,
    quantile: float = 0.97,
):
    """
    Compute an anomaly threshold from healthy scores.
    """

    return np.quantile(
        healthy_scores,
        quantile
    )


# ==========================================================
# Streaming Utilities
# ==========================================================

def blockify(
    X,
    y,
    block_length=11,
):
    """
    Convert windows into contiguous blocks.
    """

    n = (len(X) // block_length) * block_length

    X_blocks = X[:n].reshape(
        -1,
        block_length,
        X.shape[1]
    )

    y_blocks = y[:n].reshape(
        -1,
        block_length
    )

    return X_blocks, y_blocks


def interleave_blocks(
    X_blocks_A,
    y_blocks_A,
    X_blocks_B,
    y_blocks_B,
):
    """
    Uniformly interleave two streams of blocks.
    """

    n_A = len(X_blocks_A)
    n_B = len(X_blocks_B)

    output_X = []
    output_y = []

    i_A = 0
    i_B = 0

    while i_A < n_A or i_B < n_B:

        progress_A = i_A / max(n_A, 1)
        progress_B = i_B / max(n_B, 1)

        take_A = (
            i_A < n_A
            and (
                i_B >= n_B
                or progress_A <= progress_B
            )
        )

        if take_A:

            output_X.append(
                X_blocks_A[i_A]
            )

            output_y.append(
                y_blocks_A[i_A]
            )

            i_A += 1

        else:

            output_X.append(
                X_blocks_B[i_B]
            )

            output_y.append(
                y_blocks_B[i_B]
            )

            i_B += 1

    X_stream = np.concatenate(
        output_X,
        axis=0
    ).reshape(
        -1,
        X_blocks_A.shape[2]
    )

    y_stream = np.concatenate(
        output_y,
        axis=0
    ).reshape(-1)

    return X_stream, y_stream


# ==========================================================
# Block-Level Scores
# ==========================================================

def contiguous_block_scores(
    scores,
    block_length=11,
    reduction="sum",
):
    """
    Aggregate anomaly scores over blocks.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    n_blocks = len(scores) // block_length

    scores = scores[
        : n_blocks * block_length
    ]

    blocks = scores.reshape(
        n_blocks,
        block_length
    )

    if reduction == "sum":

        return blocks.sum(axis=1)

    elif reduction == "mean":

        return blocks.mean(axis=1)

    raise ValueError(
        "reduction must be 'sum' or 'mean'"
    )


def contiguous_block_labels(
    labels,
    block_length=11,
    rule="any",
):
    """
    Compute block-level labels.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    n_blocks = len(labels) // block_length

    labels = labels[
        : n_blocks * block_length
    ]

    blocks = labels.reshape(
        n_blocks,
        block_length
    )

    if rule == "any":

        return (
            blocks.max(axis=1) > 0
        ).astype(np.int64)

    elif rule == "majority":

        return (
            blocks.mean(axis=1) >= 0.5
        ).astype(np.int64)

    raise ValueError(
        "rule must be 'any' or 'majority'"
    )