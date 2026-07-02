# src/encoder.py

import torch
import torch.nn as nn


class BaseEncoderCNN(nn.Module):
    """
    1D CNN encoder used by TS-TCC.

    Input:
        (B, L) or (B, C, L)

    Output:
        (B, final_out_channels, T)
    """

    def __init__(
        self,
        input_channels: int = 1,
        kernel_size: int = 8,
        stride: int = 1,
        dropout: float = 0.1,
        final_out_channels: int = 128,
    ):
        super().__init__()

        self.final_out_channels = final_out_channels

        self.conv_block1 = self._conv_block(
            input_channels,
            32,
            kernel_size,
            stride=stride,
            dropout=dropout,
        )

        self.conv_block2 = self._conv_block(
            32,
            64,
            max(kernel_size // 2, 3),
            dropout=dropout,
        )

        self.conv_block3 = self._conv_block(
            64,
            128,
            3,
            dilation=2,
            dropout=dropout,
        )

        self.conv_block4 = self._conv_block(
            128,
            final_out_channels,
            3,
            dilation=4,
            pool=False,
            dropout=dropout,
        )

    @staticmethod
    def _conv_block(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pool=True,
        dilation=1,
        dropout=0.1,
    ):

        padding = (kernel_size // 2) * dilation

        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(
                nn.MaxPool1d(
                    kernel_size=2,
                    stride=2,
                )
            )

        layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Accept both (B, L) and (B, C, L)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        return x