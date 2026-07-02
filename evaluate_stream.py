# evaluate_stream.py

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.model import TSTCC
from src.embeddings import extract_context_vector
from src.anomaly import (
    fit_mahalanobis,
    mahalanobis_scores,
    blockify,
    interleave_blocks,
    contiguous_block_scores,
    contiguous_block_labels,
)


# ==========================================================
# Configuration
# ==========================================================

BLOCK_SIZE = 5
REDUCTION = "mean"          # "sum" or "mean"
LABEL_RULE = "majority"    # "any" or "majority"

BATCH_SIZE = 64
T_FIXED = 90

THRESHOLD_STD_FACTOR = 4.0
QUANTILE = 0.99


# ==========================================================
# Load data
# ==========================================================
# Expected files:
#
# healthy_train_segments.npy
# X_test.npy
# y_test_bin.npy
#

healthy_train_segments = np.load(
    "data/processed/healthy_train_segments.npy"
)

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test_bin = np.load(
    "data/processed/y_test_bin.npy"
)


# ==========================================================
# Separate healthy and damaged windows
# ==========================================================

X_healthy = X_test[y_test_bin == 0]
y_healthy = y_test_bin[y_test_bin == 0]

X_damaged = X_test[y_test_bin == 1]
y_damaged = y_test_bin[y_test_bin == 1]


print(f"Healthy windows: {len(X_healthy)}")
print(f"Damaged windows: {len(X_damaged)}")


# ==========================================================
# Create blocks
# ==========================================================

Xh_blocks, yh_blocks = blockify(
    X_healthy,
    y_healthy,
    block_length=BLOCK_SIZE,
)

Xd_blocks, yd_blocks = blockify(
    X_damaged,
    y_damaged,
    block_length=BLOCK_SIZE,
)


# ==========================================================
# Uniform interleaving
# ==========================================================

X_stream, y_stream = interleave_blocks(
    Xh_blocks,
    yh_blocks,
    Xd_blocks,
    yd_blocks,
)


print("Streaming test set:")
print("X_stream:", X_stream.shape)
print("y_stream:", y_stream.shape)


# ==========================================================
# Device
# ==========================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# Load model
# ==========================================================

model = TSTCC(
    device=device,
    batch_size=BATCH_SIZE,
)

model.load_state_dict(
    torch.load(
        "checkpoints/tstcc_pretrained.pt",
        map_location=device,
    )
)

model.eval()


# ==========================================================
# Extract embeddings
# ==========================================================

Z_train = extract_context_vector(
    model,
    healthy_train_segments,
    batch_size=BATCH_SIZE,
    device=device,
    t_fixed=T_FIXED,
)

Z_stream = extract_context_vector(
    model,
    X_stream,
    batch_size=BATCH_SIZE,
    device=device,
    t_fixed=T_FIXED,
)


print("Z_train :", Z_train.shape)
print("Z_stream:", Z_stream.shape)


# ==========================================================
# Mahalanobis model
# ==========================================================

mu, L = fit_mahalanobis(
    Z_train,
    eps=1e-3,
)


scores_train = mahalanobis_scores(
    Z_train,
    mu,
    L,
)

scores_stream = mahalanobis_scores(
    Z_stream,
    mu,
    L,
)


# ==========================================================
# Block aggregation
# ==========================================================

scores_train_blocks = contiguous_block_scores(
    scores_train,
    block_length=BLOCK_SIZE,
    reduction=REDUCTION,
)

scores_test_blocks = contiguous_block_scores(
    scores_stream,
    block_length=BLOCK_SIZE,
    reduction=REDUCTION,
)


y_test_blocks = contiguous_block_labels(
    y_stream,
    block_length=BLOCK_SIZE,
    rule=LABEL_RULE,
)


# ==========================================================
# Threshold
# ==========================================================
# Option 1:
# threshold = np.quantile(
#     scores_train_blocks,
#     QUANTILE
# )

# Option 2 (recommended):
threshold = (
    scores_train_blocks.mean()
    + THRESHOLD_STD_FACTOR
    * scores_train_blocks.std()
)


print(
    f"\nThreshold = {threshold:.4f}"
)


# ==========================================================
# Predictions
# ==========================================================

y_pred_blocks = (
    scores_test_blocks > threshold
).astype(np.int64)


# ==========================================================
# Metrics
# ==========================================================

cm = confusion_matrix(
    y_test_blocks,
    y_pred_blocks,
)

tn, fp, fn, tp = cm.ravel()


print("\nBlock-level confusion matrix:")
print(cm)


print(
    f"FPR: {fp/(fp+tn+1e-12):.4f}"
)

print(
    f"TPR: {tp/(tp+fn+1e-12):.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test_blocks, scores_test_blocks):.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision_score(y_test_blocks, scores_test_blocks):.4f}"
)


# ==========================================================
# Confusion Matrix
# ==========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "Damaged",
    ],
)

disp.plot(
    cmap=plt.cm.Blues,
    values_format="d",
)

plt.title(
    f"Online Block CM "
    f"(K={BLOCK_SIZE}, {REDUCTION})"
)

plt.show()


# ==========================================================
# Histogram
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    scores_test_blocks[
        y_test_blocks == 0
    ],
    bins=10,
    alpha=0.6,
    label="Healthy blocks",
)

plt.hist(
    scores_test_blocks[
        y_test_blocks == 1
    ],
    bins=100,
    alpha=0.6,
    label="Damaged blocks",
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Threshold",
)

plt.xlabel("Block anomaly score")
plt.ylabel("Count")

plt.title(
    f"Online Block Scores "
    f"(K={BLOCK_SIZE}, {REDUCTION})"
)

plt.legend()

plt.show()