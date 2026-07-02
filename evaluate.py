# evaluate.py

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
    compute_threshold,
)


# ==========================================================
# Load data
# ==========================================================

X_test = np.load("data/processed/X_test.npy")
y_test_bin = np.load("data/processed/y_test_bin.npy")

healthy_train_segments = np.load(
    "data/processed/healthy_train_segments.npy"
)


# ==========================================================
# Load model
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TSTCC(
    device=device,
    batch_size=64,
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
    batch_size=64,
    device=device,
    t_fixed=90,
)

Z_test = extract_context_vector(
    model,
    X_test,
    batch_size=64,
    device=device,
    t_fixed=90,
)


# ==========================================================
# Mahalanobis model
# ==========================================================

mu, L = fit_mahalanobis(Z_train)

scores_train = mahalanobis_scores(
    Z_train,
    mu,
    L,
)

scores_test = mahalanobis_scores(
    Z_test,
    mu,
    L,
)


# ==========================================================
# Threshold
# ==========================================================

threshold = compute_threshold(
    scores_train,
    quantile=0.97,
)

print(f"Threshold: {threshold:.4f}")


# ==========================================================
# Predictions
# ==========================================================

y_pred = (
    scores_test > threshold
).astype(np.int64)


# ==========================================================
# Metrics
# ==========================================================

cm = confusion_matrix(
    y_test_bin,
    y_pred,
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print(
    f"FPR: {fp / (fp + tn + 1e-12):.4f}"
)

print(
    f"TPR: {tp / (tp + fn + 1e-12):.4f}"
)

print(
    f"ROC-AUC: {roc_auc_score(y_test_bin, scores_test):.4f}"
)

print(
    f"AP: {average_precision_score(y_test_bin, scores_test):.4f}"
)


# ==========================================================
# Confusion Matrix
# ==========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Damaged"],
)

disp.plot(
    cmap=plt.cm.Blues,
    values_format="d",
)

plt.show()


# ==========================================================
# Score Histograms
# ==========================================================

plt.figure()

plt.hist(
    scores_test[y_test_bin == 0],
    bins=10,
    alpha=0.6,
    label="Healthy",
)

plt.hist(
    scores_test[y_test_bin == 1],
    bins=100,
    alpha=0.6,
    label="Damaged",
)

plt.axvline(
    threshold,
    linestyle="--",
)

plt.xlabel("Mahalanobis Score")
plt.ylabel("Count")

plt.legend()

plt.show()