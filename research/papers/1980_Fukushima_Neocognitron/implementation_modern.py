"""
Fukushima (1980) – Neocognitron
MODERN ADAPTATION (Pedagogical)

Characteristics:
- S-cell response: weighted sum with normalisation
- C-cell response: average pooling over a local region
- Demonstrates the architecture with NumPy
"""

import numpy as np

class NeocognitronModern:
    def __init__(self, num_filters=3, filter_size=3, pool_size=2):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def s_cell_response(self, region, filter_weights):
        numerator = np.sum(region * filter_weights)
        denominator = np.sqrt(np.sum(filter_weights**2) + np.sum(region**2) + 1e-8)
        return numerator / denominator

    def forward(self, X):
        _, h, w = X.shape
        k = self.filter_size
        h_out = h - k + 1
        w_out = w - k + 1
        s_layer = np.zeros((self.num_filters, h_out, w_out))
        for f in range(self.num_filters):
            for i in range(h_out):
                for j in range(w_out):
                    region = X[:, i:i+k, j:j+k]
                    s_layer[f, i, j] = self.s_cell_response(region, self.filters[f])
        # C-cell pooling
        _, h, w = s_layer.shape
        h_out = h // self.pool_size
        w_out = w // self.pool_size
        c_layer = np.zeros((self.num_filters, h_out, w_out))
        for f in range(self.num_filters):
            for i in range(h_out):
                for j in range(w_out):
                    region = s_layer[f, i*self.pool_size:(i+1)*self.pool_size,
                                     j*self.pool_size:(j+1)*self.pool_size]
                    c_layer[f, i, j] = np.mean(region)
        return c_layer

if __name__ == "__main__":
    print("=== Neocognitron Modern Demo ===")
    np.random.seed(42)
    X = np.random.randn(1, 16, 16)
    net = NeocognitronModern(num_filters=3, filter_size=3, pool_size=2)
    out = net.forward(X)
    print(f"Output shape: {out.shape}")
