# src/trainer.py

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import WindowsDataset
from src.augmentations import weak_view, strong_view
from src.model import TSTCC


def pretrain_tstcc(
    healthy_train_segments: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    tc_timesteps: int = 16,
    tc_hidden_dim: int = 256,
    final_out_channels: int = 128,
    lambda_cc: float = 0.7,
    cc_temperature: float = 0.2,
    verbose: bool = True,
):
    """
    Pretrain TS-TCC on healthy data only.

    Parameters
    ----------
    healthy_train_segments : np.ndarray
        Shape (N, L)

    Returns
    -------
    model : TSTCC
        Pretrained model.
    """

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==================================================
    # Dataset
    # ==================================================
    dataset = WindowsDataset(
        healthy_train_segments,
        y=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ==================================================
    # Model
    # ==================================================
    model = TSTCC(
        device=device,
        final_out_channels=final_out_channels,
        tc_timesteps=tc_timesteps,
        tc_hidden_dim=tc_hidden_dim,
        cc_temperature=cc_temperature,
        batch_size=batch_size,
        lambda_cc=lambda_cc,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # ==================================================
    # Training loop
    # ==================================================
    model.train()

    for epoch in range(1, epochs + 1):

        epoch_loss = 0.0
        epoch_nce1 = 0.0
        epoch_nce2 = 0.0
        epoch_cc = 0.0

        n_batches = 0

        for batch in dataloader:

            batch = batch.to(device)

            # ------------------------------------------
            # Generate two augmented views
            # ------------------------------------------
            x_weak = weak_view(batch)
            x_strong = strong_view(batch)

            # ------------------------------------------
            # Forward
            # ------------------------------------------
            loss, logs = model(
                x_weak,
                x_strong,
            )

            # ------------------------------------------
            # Backpropagation
            # ------------------------------------------
            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------
            # Logging
            # ------------------------------------------
            epoch_loss += loss.item()
            epoch_nce1 += logs["nce1"].item()
            epoch_nce2 += logs["nce2"].item()
            epoch_cc += logs["cc"].item()

            n_batches += 1

        # ==================================================
        # Epoch statistics
        # ==================================================
        avg_loss = epoch_loss / n_batches
        avg_nce1 = epoch_nce1 / n_batches
        avg_nce2 = epoch_nce2 / n_batches
        avg_cc = epoch_cc / n_batches

        avg_nce = avg_nce1 + avg_nce2

        if verbose and (epoch % 5 == 0 or epoch == 1):

            reconstructed_loss = (
                avg_nce +
                lambda_cc * avg_cc
            )

            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"Loss={avg_loss:.4f} | "
                f"NCE={avg_nce:.4f} "
                f"(NCE1={avg_nce1:.4f}, "
                f"NCE2={avg_nce2:.4f}) | "
                f"CC={avg_cc:.4f} | "
                f"Reconstructed={reconstructed_loss:.4f}"
            )

    return model