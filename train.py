# train.py

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.datasets import create_segments
from src.trainer import pretrain_tstcc


# ==========================================================
# Configuration
# ==========================================================

SEG_LEN = 2048
STRIDE_PRE = 2048

EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3

TC_TIMESTEPS = 32
TC_HIDDEN_DIM = 128
FINAL_OUT_CHANNELS = 128

LAMBDA_CC = 0.6
CC_TEMPERATURE = 0.3


# ==========================================================
# Load data
# ==========================================================
# Replace this with your own loading function

X_normalized = np.load("data/processed/X_normalized.npy")
labels = np.load("data/processed/labels.npy")


# ==========================================================
# Select one channel
# ==========================================================

X_ch0 = X_normalized[:, :, 0].astype(np.float32)
y = labels.astype(np.int64)


# ==========================================================
# Healthy train set
# ==========================================================

healthy_series = X_ch0[y == 0]

healthy_train, healthy_val = train_test_split(
    healthy_series,
    test_size=0.11,
    random_state=42,
    shuffle=True,
)

healthy_train_segments, _ = create_segments(
    healthy_train,
    np.zeros(len(healthy_train), dtype=np.int64),
    segment_length=SEG_LEN,
    stride=STRIDE_PRE,
)

print("Healthy train segments:", healthy_train_segments.shape)


# ==========================================================
# Train
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = pretrain_tstcc(
    healthy_train_segments,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=device,
    tc_timesteps=TC_TIMESTEPS,
    tc_hidden_dim=TC_HIDDEN_DIM,
    final_out_channels=FINAL_OUT_CHANNELS,
    lambda_cc=LAMBDA_CC,
    cc_temperature=CC_TEMPERATURE,
)


# ==========================================================
# Save model
# ==========================================================

torch.save(
    model.state_dict(),
    "checkpoints/tstcc_pretrained.pt"
)

print("Model saved successfully.")