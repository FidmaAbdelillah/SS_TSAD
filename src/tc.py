# src/tc.py

import numpy as np
import torch
import torch.nn as nn

from src.transformer import SeqTransformer


class TC(nn.Module):
    """
    Temporal Contrasting module of TS-TCC.

    It learns temporal dependencies by predicting
    future latent representations from a context vector.
    """

    def __init__(
        self,
        device: str,
        final_out_channels: int = 128,
        timesteps: int = 8,
        hidden_dim: int = 256,
        proj_dim: int = 32,
        tr_depth: int = 8,
        tr_heads: int = 4,
        tr_mlp_dim: int = 256,
    ):
        super().__init__()

        self.device = device
        self.num_channels = final_out_channels
        self.timestep = timesteps

        # Future prediction heads
        self.Wk = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_channels)
            for _ in range(self.timestep)
        ])

        self.log_softmax = nn.LogSoftmax(dim=1)

        # Projection head for contextual contrastive loss
        hidden_proj = max(final_out_channels // 2, 8)

        if proj_dim is None:
            proj_dim = max(final_out_channels // 4, 8)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_proj),
            nn.BatchNorm1d(hidden_proj),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_proj, proj_dim),
        )

        # Sequence Transformer
        self.seq_transformer = SeqTransformer(
            patch_size=self.num_channels,
            dim=hidden_dim,
            depth=tr_depth,
            heads=tr_heads,
            mlp_dim=tr_mlp_dim,
            dropout=0.1,
        )

    def forward(
        self,
        features_aug1: torch.Tensor,
        features_aug2: torch.Tensor,
    ):
        """
        Parameters
        ----------
        features_aug1 : Tensor
            Shape (B, C, T)

        features_aug2 : Tensor
            Shape (B, C, T)

        Returns
        -------
        nce_loss : Tensor

        projected_context : Tensor
            Shape (B, proj_dim)
        """

        # Convert (B, C, T) -> (B, T, C)
        z_aug1 = features_aug1.transpose(1, 2)
        z_aug2 = features_aug2.transpose(1, 2)

        batch_size, sequence_length, _ = z_aug1.shape

        # Random temporal position
        t_sample = torch.randint(
            sequence_length - self.timestep,
            size=(1,),
            device=self.device,
        ).long()

        # Future target representations
        future_targets = torch.empty(
            self.timestep,
            batch_size,
            self.num_channels,
            device=self.device,
        )

        for i in range(1, self.timestep + 1):

            future_targets[i - 1] = z_aug2[
                :,
                t_sample + i,
                :
            ].view(
                batch_size,
                self.num_channels,
            )

        # Context sequence
        forward_sequence = z_aug1[:, :t_sample + 1, :]

        # Context vector
        context_vector = self.seq_transformer(
            forward_sequence
        )

        # Predict future representations
        predictions = torch.empty(
            self.timestep,
            batch_size,
            self.num_channels,
            device=self.device,
        )

        for i in range(self.timestep):

            predictions[i] = self.Wk[i](
                context_vector
            )

        # NCE loss
        nce_loss = 0.0

        for i in range(self.timestep):

            scores = torch.mm(
                future_targets[i],
                predictions[i].transpose(0, 1)
            )

            nce_loss += torch.sum(
                torch.diag(
                    self.log_softmax(scores)
                )
            )

        nce_loss /= (-batch_size * self.timestep)

        # Projection head output
        projected_context = self.projection_head(
            context_vector
        )

        return nce_loss, projected_context