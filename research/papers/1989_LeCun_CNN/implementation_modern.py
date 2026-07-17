"""
LeCun (1989) – CNN Implementation
MODERN ADAPTATION (Pedagogical)

Characteristics:
- Activation: ReLU (modern best practice).
- Pooling: Max pooling (vs. average).
- Optimiser: Adam with adaptive learning rate.
- Initialisation: He initialisation (Xavier variant).
"""

import numpy as np

class Conv2D_Modern:
    def __init__(self, input_depth, output_depth, filter_size, lr=0.001):
        self.lr = lr
        # He initialisation (modern)
        self.filters = np.random.randn(output_depth, input_depth, filter_size, filter_size) * np.sqrt(2.0 / (input_depth * filter_size * filter_size))
        self.bias = np.zeros(output_depth)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        # Convolution followed by ReLU
        self.X = X
        batch, in_depth, h, w = X.shape
        out_depth, _, k, _ = self.filters.shape
        h_out = h - k + 1
        w_out = w - k + 1
        self.output = np.zeros((batch, out_depth, h_out, w_out))

        for b in range(batch):
            for d_out in range(out_depth):
                for i in range(h_out):
                    for j in range(w_out):
                        rf = X[b, :, i:i+k, j:j+k]
                        self.output[b, d_out, i, j] = np.sum(rf * self.filters[d_out]) + self.bias[d_out]
        self.output = self.relu(self.output)
        return self.output

class MaxPool2D_Modern:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        self.X = X
        batch, depth, h, w = X.shape
        h_out = h // self.pool_size
        w_out = w // self.pool_size
        self.output = np.zeros((batch, depth, h_out, w_out))
        self.max_indices = {}

        for b in range(batch):
            for d in range(depth):
                for i in range(h_out):
                    for j in range(w_out):
                        region = X[b, d, i*self.pool_size:(i+1)*self.pool_size,
                                   j*self.pool_size:(j+1)*self.pool_size]
                        max_val = np.max(region)
                        self.output[b, d, i, j] = max_val
                        # Store index for backward pass
                        idx = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[(b, d, i, j)] = idx
        return self.output

    def backward(self, grad_output):
        # Max pooling backward pass (only propagate to the max location)
        pass