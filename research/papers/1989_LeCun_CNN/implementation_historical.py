"""
LeCun (1989) – CNN Implementation
HISTORICAL IMPLEMENTATION (Faithful to the Original)

Characteristics:
- Activation: Sigmoid (as used in the 1989 paper).
- Pooling: Average pooling (subsampling).
- Optimiser: Mini-batch SGD with a fixed learning rate (no momentum, no Adam).
- Initialisation: Small uniform random weights (consistent with the period).
- Normalization: None (batch norm did not exist).

This implementation prioritises historical accuracy over modern performance.
"""

import numpy as np

class Conv2D_Historical:
    """2D Convolutional layer with sigmoid activation and learnable filters."""
    def __init__(self, input_depth, output_depth, filter_size, lr=0.01):
        self.lr = lr
        # Small uniform initialisation (historical practice)
        self.filters = np.random.uniform(-0.1, 0.1, (output_depth, input_depth, filter_size, filter_size))
        self.bias = np.zeros(output_depth)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        """Forward pass: convolution followed by sigmoid activation."""
        self.X = X
        batch, in_depth, h, w = X.shape
        out_depth, _, k, _ = self.filters.shape
        h_out = h - k + 1
        w_out = w - k + 1

        self.z = np.zeros((batch, out_depth, h_out, w_out))
        self.output = np.zeros((batch, out_depth, h_out, w_out))

        for b in range(batch):
            for d_out in range(out_depth):
                for i in range(h_out):
                    for j in range(w_out):
                        rf = X[b, :, i:i+k, j:j+k]
                        self.z[b, d_out, i, j] = np.sum(rf * self.filters[d_out]) + self.bias[d_out]

        self.output = self.sigmoid(self.z)
        return self.output

    def backward(self, grad_output):
        """Backward pass: compute gradients and update weights."""
        batch, out_depth, h_out, w_out = grad_output.shape
        _, in_depth, h, w = self.X.shape
        k = self.filters.shape[2]

        # Gradient through sigmoid activation
        grad_z = grad_output * self.sigmoid_derivative(self.output)

        grad_input = np.zeros_like(self.X)
        grad_filters = np.zeros_like(self.filters)

        for b in range(batch):
            for d_out in range(out_depth):
                for i in range(h_out):
                    for j in range(w_out):
                        grad = grad_z[b, d_out, i, j]
                        # Gradient w.r.t input
                        grad_input[b, :, i:i+k, j:j+k] += grad * self.filters[d_out]
                        # Gradient w.r.t filters
                        grad_filters[d_out] += grad * self.X[b, :, i:i+k, j:j+k]

        # Update weights (mini-batch SGD)
        self.filters -= self.lr * grad_filters / batch
        self.bias -= self.lr * np.sum(grad_z, axis=(0, 2, 3)) / batch

        return grad_input


class AvgPool2D_Historical:
    """Average pooling layer (subsampling)."""
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        self.X = X
        batch, depth, h, w = X.shape
        h_out = h // self.pool_size
        w_out = w // self.pool_size
        self.output = np.zeros((batch, depth, h_out, w_out))

        for b in range(batch):
            for d in range(depth):
                for i in range(h_out):
                    for j in range(w_out):
                        region = X[b, d, i*self.pool_size:(i+1)*self.pool_size,
                                   j*self.pool_size:(j+1)*self.pool_size]
                        self.output[b, d, i, j] = np.mean(region)
        return self.output

    def backward(self, grad_output):
        """Backward pass: distribute error evenly across the pooling region."""
        batch, depth, h_out, w_out = grad_output.shape
        _, _, h, w = self.X.shape
        grad_input = np.zeros_like(self.X)
        pool_area = self.pool_size * self.pool_size

        for b in range(batch):
            for d in range(depth):
                for i in range(h_out):
                    for j in range(w_out):
                        grad = grad_output[b, d, i, j] / pool_area
                        grad_input[b, d, i*self.pool_size:(i+1)*self.pool_size,
                                   j*self.pool_size:(j+1)*self.pool_size] = grad
        return grad_input


# === Historical XOR Demo (2-layer CNN) ===
if __name__ == "__main__":
    np.random.seed(42)
    print("=== Historical CNN (Sigmoid, Average Pooling, SGD) ===")

    # Dummy data: 4 samples of 4x4 binary images
    X = np.random.randn(4, 1, 4, 4)
    y = np.array([[0], [1], [1], [0]])  # XOR-like pattern (conceptually)

    conv = Conv2D_Historical(input_depth=1, output_depth=1, filter_size=2, lr=0.01)
    pool = AvgPool2D_Historical()

    # Forward pass
    out_conv = conv.forward(X)   # (4, 1, 3, 3)
    out_pool = pool.forward(out_conv)  # (4, 1, 1, 1)

    # Flatten and simple linear layer (for demonstration)
    flat = out_pool.reshape(4, -1)
    W = np.random.randn(1, 1) * 0.1
    pred = flat @ W

    print(f"Forward pass successful. Output shape: {out_pool.shape}")
    print(f"Predictions (raw): {pred.flatten()}")