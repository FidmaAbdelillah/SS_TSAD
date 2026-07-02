# src/model.py

import torch
import torch.nn as nn

from src.encoder import BaseEncoderCNN
from src.tc import TC
from src.losses import NTXentLoss


class TSTCC(nn.Module):
    """
    TS-TCC model.

    Combines:

    - CNN encoder
    - Temporal Contrasting (TC)
    - Contextual Contrasting (NT-Xent)
    """

    def __init__(
        self,
        device: str,
        input_channels: int = 1,
        enc_kernel_size: int = 9,
        enc_stride: int = 1,
        enc_dropout: float = 0.1,
        final_out_channels: int = 128,
        tc_timesteps: int = 32,
        tc_hidden_dim: int = 128,
        cc_temperature: float = 0.2,
        batch_size: int = 64,
        lambda_cc: float = 0.7,
    ):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.lambda_cc = lambda_cc

        # ==================================================
        # CNN Encoder
        # ==================================================
        self.encoder = BaseEncoderCNN(
            input_channels=input_channels,
            kernel_size=enc_kernel_size,
            stride=enc_stride,
            dropout=enc_dropout,
            final_out_channels=final_out_channels,
        )

        # ==================================================
        # Temporal Contrasting Module
        # ==================================================
        self.TC = TC(
            device=device,
            final_out_channels=final_out_channels,
            timesteps=tc_timesteps,
            hidden_dim=tc_hidden_dim,
            proj_dim=final_out_channels // 4,
            tr_depth=4,
            tr_heads=16,
            tr_mlp_dim=128,
        )

        # ==================================================
        # Contextual Contrastive Loss
        # ==================================================
        self.CC = NTXentLoss(
            device=device,
            batch_size=batch_size,
            temperature=cc_temperature,
            use_cosine_similarity=True,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x1 : Tensor
            First augmented batch.

        x2 : Tensor
            Second augmented batch.

        Returns
        -------
        loss : Tensor

        logs : dict
            Dictionary containing:

            - nce1
            - nce2
            - cc
        """

        # ==================================================
        # Encoder
        # ==================================================
        features_1 = self.encoder(x1)
        features_2 = self.encoder(x2)

        # ==================================================
        # Temporal Contrasting
        # ==================================================
        nce_1, projection_1 = self.TC(
            features_1,
            features_2,
        )

        nce_2, projection_2 = self.TC(
            features_2,
            features_1,
        )

        # ==================================================
        # Contextual Contrasting
        # ==================================================
        cc_loss = self.CC(
            projection_1,
            projection_2,
        )

        # ==================================================
        # Total Loss
        # ==================================================
        total_loss = (
            nce_1 +
            nce_2 +
            self.lambda_cc * cc_loss
        )

        logs = {
            "nce1": nce_1.detach(),
            "nce2": nce_2.detach(),
            "cc": cc_loss.detach(),
        }

        return total_loss, logs