# src/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class Residual(nn.Module):
    """
    Residual connection wrapper.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Layer normalization before a module.
    """

    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Transformer feed-forward block.
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention.
    """

    def __init__(
        self,
        dim,
        heads=8,
        dropout=0.0
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(
            dim,
            dim * 3,
            bias=False
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, x, mask=None):

        batch_size, seq_len, dim = x.shape
        heads = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b n (h d) -> b h n d",
                h=heads
            ),
            qkv
        )

        attention_scores = (
            torch.einsum(
                "bhid,bhjd->bhij",
                q,
                k
            )
            * self.scale
        )

        if mask is not None:

            mask = F.pad(
                mask.flatten(1),
                (1, 0),
                value=True
            )

            mask = mask[:, None, :] * mask[:, :, None]

            attention_scores.masked_fill_(
                ~mask,
                float("-inf")
            )

            del mask

        attention_weights = attention_scores.softmax(dim=-1)

        out = torch.einsum(
            "bhij,bhjd->bhid",
            attention_weights,
            v
        )

        out = rearrange(
            out,
            "b h n d -> b n (h d)"
        )

        return self.to_out(out)


class Transformer(nn.Module):
    """
    Standard Transformer encoder.
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(depth):

            self.layers.append(
                nn.ModuleList([
                    Residual(
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dropout=dropout
                            )
                        )
                    ),

                    Residual(
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout
                            )
                        )
                    )
                ])
            )

    def forward(self, x, mask=None):

        for attention_layer, feedforward_layer in self.layers:

            x = attention_layer(x, mask=mask)
            x = feedforward_layer(x)

        return x


class SeqTransformer(nn.Module):
    """
    Sequence Transformer used in TS-TCC.

    Converts temporal features into a context vector.
    """

    def __init__(
        self,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=1,
        dropout=0.1
    ):
        super().__init__()

        patch_dim = channels * patch_size

        self.patch_to_embedding = nn.Linear(
            patch_dim,
            dim
        )

        self.context_token = nn.Parameter(
            torch.randn(1, 1, dim)
        )

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout
        )

    def forward(self, forward_sequence):

        x = self.patch_to_embedding(forward_sequence)

        batch_size, seq_len, _ = x.shape

        context_tokens = repeat(
            self.context_token,
            "() n d -> b n d",
            b=batch_size
        )

        x = torch.cat(
            (context_tokens, x),
            dim=1
        )

        x = self.transformer(x)

        context_vector = x[:, 0]

        return context_vector