# Self-Supervised Anomaly Detection for Structural Health Monitoring

This repository implements a self-supervised anomaly detection framework based on TS-TCC (Temporal and Contextual Contrasting) for vibration-based Structural Health Monitoring (SHM).

The framework is trained only on healthy data and uses Mahalanobis distance in the latent space for damage detection.

---

## Project Structure

```text
SSAD_TSTCC/

├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── datasets.py
│   ├── augmentations.py
│   ├── encoder.py
│   ├── transformer.py
│   ├── losses.py
│   ├── tc.py
│   ├── model.py
│   ├── trainer.py
│   ├── embeddings.py
│   └── anomaly.py
│
├── notebooks/
│   └── example_workflow.ipynb
│
├── checkpoints/
├── results/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is not included in this repository.

Expected files:

```text
data/processed/

X_normalized.npy
labels.npy
healthy_train_segments.npy
X_test.npy
y_test_bin.npy
```

---

## Training

Train the TS-TCC model:

```bash
python train.py
```

The trained model will be saved in:

```text
checkpoints/tstcc_pretrained.pt
```

---

## Evaluation

Evaluate anomaly detection performance:

```bash
python evaluate.py
```

The evaluation includes:

- Mahalanobis distance
- Threshold selection
- Confusion matrix
- ROC-AUC
- Average Precision
- Score histograms

---

## Main Hyperparameters

| Parameter | Value |
|-----------|--------|
| Segment length | 2048 |
| Batch size | 64 |
| Epochs | 200 |
| Learning rate | 1e-3 |
| TC timesteps | 32 |
| Hidden dimension | 128 |
| Lambda CC | 0.6 |
| Temperature | 0.3 |

---

## Authors

Mohamed Abdelillah Fidma  
Université Gustave Eiffel (EMGCU)