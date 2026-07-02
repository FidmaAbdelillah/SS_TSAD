import numpy as np
from src.datasets import WindowsDataset, create_segments


X = np.random.randn(10, 2048)
y = np.zeros(10)

dataset = WindowsDataset(X, y)

print(len(dataset))
print(dataset[0][0].shape)


X_seg, y_seg = create_segments(
    X,
    y,
    segment_length=512,
    stride=512
)

print(X_seg.shape)
print(y_seg.shape)