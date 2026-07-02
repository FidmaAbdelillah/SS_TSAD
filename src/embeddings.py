# src/embeddings.py

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import WindowsDataset


@torch.no_grad()
def extract_context_vector(
    model,
    X: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
    t_fixed: int | None = None,
):
    """
    Extract context vectors (c_t) from a trained TS-TCC model.

    Parameters
    ----------
    model : TSTCC
        Trained TS-TCC model.

    X : np.ndarray
        Input windows of shape (N, L).

    batch_size : int
        Inference batch size.

    device : str, optional
        Computation device.

    t_fixed : int, optional
        Fixed temporal position used to extract c_t.
        If None, a random valid position is selected.

    Returns
    -------
    np.ndarray
        Context vectors of shape (N, hidden_dim).
    """

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.eval()
    model.to(device)

    dataset = WindowsDataset(X, y=None)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    embeddings = []

    for batch in dataloader:

        batch = batch.to(device)

        # ------------------------------------------
        # Encoder features
        # (B, C, T)
        # ------------------------------------------
        features = model.encoder(batch)

        # (B, T, C)
        features = features.transpose(1, 2)

        batch_size_current, T, C = features.shape
        K = model.TC.timestep

        # ------------------------------------------
        # Select temporal position
        # ------------------------------------------
        if t_fixed is None:

            t = torch.randint(
                T - K,
                size=(1,),
                device=device,
            ).long()

        else:

            t_value = max(
                0,
                min(int(t_fixed), T - K - 1)
            )

            t = torch.tensor(
                [t_value],
                device=device,
            ).long()

        # ------------------------------------------
        # Build context sequence
        # ------------------------------------------
        forward_sequence = features[:, :t + 1, :]

        # ------------------------------------------
        # Extract context vector
        # ------------------------------------------
        context_vector = model.TC.seq_transformer(
            forward_sequence
        )

        embeddings.append(
            context_vector.cpu()
        )

    return torch.cat(
        embeddings,
        dim=0,
    ).numpy()