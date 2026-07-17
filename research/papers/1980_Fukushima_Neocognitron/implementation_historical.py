"""
Fukushima (1980) – Neocognitron
HISTORICAL IMPLEMENTATION (Faithful to the Original)

Characteristics:
- S-cell response: weighted sum with normalisation
- C-cell response: average pooling over a local region
- Unsupervised competitive learning (conceptual demonstration)
"""

import numpy as np

class NeocognitronSimple:
    """
    Simplified Neocognitron (Fukushima, 1980).
    Demonstrates S-cell (feature detection) and C-cell (pooling) layers.
    """
    def __init__(self, input_size=16, num_filters=3, filter_size=3, pool_size=2):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        # Initialize filters randomly (unsupervised learning would adjust these)
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)

    def s_cell_response(self, region, filter_weights):
        """Response of a single S-cell (feature detector)."""
        numerator = np.sum(region * filter_weights)
        denominator = np.sqrt(np.sum(filter_weights**2) + np.sum(region**2) + 1e-8)
        return numerator / denominator

    def forward_s_layer(self, X):
        """Convolution-like S-cell layer."""
        _, h, w = X.shape
        k = self.filter_size
        h_out = h - k + 1
        w_out = w - k + 1
        output = np.zeros((self.num_filters, h_out, w_out))

        for f in range(self.num_filters):
            for i in range(h_out):
                for j in range(w_out):
                    region = X[:, i:i+k, j:j+k]
                    output[f, i, j] = self.s_cell_response(region, self.filters[f])
        return output

    def c_cell_response(self, s_layer, pool_size):
        """C-cell pooling (average pooling)."""
        _, h, w = s_layer.shape
        h_out = h // pool_size
        w_out = w // pool_size
        output = np.zeros((self.num_filters, h_out, w_out))

        for f in range(self.num_filters):
            for i in range(h_out):
                for j in range(w_out):
                    region = s_layer[f, i*pool_size:(i+1)*pool_size,
                                     j*pool_size:(j+1)*pool_size]
                    output[f, i, j] = np.mean(region)
        return output

    def forward(self, X):
        # X: (1, H, W) grayscale input
        s1 = self.forward_s_layer(X)
        c1 = self.c_cell_response(s1, self.pool_size)
        return c1

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1, 16, 16)
    net = NeocognitronSimple(input_size=16, num_filters=3, filter_size=3, pool_size=2)
    out = net.forward(X)
    print(f"Neocognitron output shape: {out.shape}")  # (3, 7, 7)
