# src/losses.py

import numpy as np
import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-Scaled Cross Entropy Loss (NT-Xent)
    used for contextual contrastive learning.
    """

    def __init__(
        self,
        device: str,
        batch_size: int,
        temperature: float = 0.2,
        use_cosine_similarity: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask_samples_from_same_repr = (
            self._get_correlated_mask().bool()
        )

        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity
        )

        self.criterion = nn.CrossEntropyLoss(
            reduction="sum"
        )

    def _get_similarity_function(
        self,
        use_cosine_similarity: bool
    ):

        if use_cosine_similarity:

            self._cosine_similarity = nn.CosineSimilarity(
                dim=-1
            )

            return self._cosine_similarity_fn

        return self._dot_similarity_fn

    def _get_correlated_mask(self):

        diag = np.eye(
            2 * self.batch_size,
            dtype=np.float32
        )

        lower_diag = np.eye(
            2 * self.batch_size,
            2 * self.batch_size,
            k=-self.batch_size,
            dtype=np.float32
        )

        upper_diag = np.eye(
            2 * self.batch_size,
            2 * self.batch_size,
            k=self.batch_size,
            dtype=np.float32
        )

        mask = diag + lower_diag + upper_diag
        mask = (1.0 - mask).astype(np.bool_)

        return torch.from_numpy(mask).to(self.device)

    @staticmethod
    def _dot_similarity_fn(x, y):

        return torch.tensordot(
            x.unsqueeze(1),
            y.T.unsqueeze(0),
            dims=2
        )

    def _cosine_similarity_fn(self, x, y):

        return self._cosine_similarity(
            x.unsqueeze(1),
            y.unsqueeze(0)
        )

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ):

        representations = torch.cat(
            [z_j, z_i],
            dim=0
        )

        similarity_matrix = self.similarity_function(
            representations,
            representations
        )

        left_positive = torch.diag(
            similarity_matrix,
            self.batch_size
        )

        right_positive = torch.diag(
            similarity_matrix,
            -self.batch_size
        )

        positives = torch.cat(
            [left_positive, right_positive]
        ).view(
            2 * self.batch_size,
            1
        )

        negatives = similarity_matrix[
            self.mask_samples_from_same_repr
        ].view(
            2 * self.batch_size,
            -1
        )

        logits = torch.cat(
            (positives, negatives),
            dim=1
        )

        logits /= self.temperature

        labels = torch.zeros(
            2 * self.batch_size,
            device=self.device
        ).long()

        loss = self.criterion(
            logits,
            labels
        )

        return loss / (2 * self.batch_size)