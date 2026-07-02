# src/datasets.py

import numpy as np
from torch.utils.data import Dataset


class WindowsDataset(Dataset):
    """
    Dataset for fixed-length time-series windows.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, L)
            N = number of samples
            L = window length

    y : np.ndarray, optional
        Shape (N,)
        Labels. If None, only X is returned.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):

        if X.ndim != 2:
            raise ValueError(
                f"Expected X to have shape (N, L), got {X.shape}"
            )

        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]

        if self.y is None:
            return x

        return x, self.y[idx]


def create_segments(
    X_TS: np.ndarray,
    y_TS: np.ndarray,
    segment_length: int,
    stride: int,
):
    """
    Split long time series into fixed-length windows.

    Parameters
    ----------
    X_TS : np.ndarray
        Shape (N_signals, signal_length)

    y_TS : np.ndarray
        Shape (N_signals,)

    segment_length : int
        Length of each segment.

    stride : int
        Sliding window stride.

    Returns
    -------
    X_segments : np.ndarray
        Shape (N_segments, segment_length)

    y_segments : np.ndarray
        Shape (N_segments,)
    """

    X_segments = []
    y_segments = []

    for i, signal in enumerate(X_TS):

        n = signal.shape[0]
        start = 0

        while start + segment_length <= n:

            X_segments.append(
                signal[start:start + segment_length]
            )

            y_segments.append(y_TS[i])

            start += stride

    return (
        np.asarray(X_segments, dtype=np.float32),
        np.asarray(y_segments, dtype=np.int64),
    )