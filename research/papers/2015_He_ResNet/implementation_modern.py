"""
ResNet (2015) – Modern Educational Implementation

This is a conceptual demonstration of the residual block structure.
Full ImageNet training with convolution layers is not implemented here
due to its complexity; the focus is on the residual connection principle.

Reference: He et al., 2015
"""

import numpy as np

class ResidualBlockModern:
    """
    Simplified residual block with identity shortcut.
    """
    def __init__(self, input_dim, hidden_dim, use_projection=False):
        self.use_projection = use_projection
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b2 = np.zeros((1, input_dim))
        if use_projection:
            self.Ws = np.random.randn(input_dim, input_dim) * 0.1
            self.bs = np.zeros((1, input_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Residual mapping: F(x)
        h = self.relu(np.dot(x, self.W1) + self.b1)
        residual = np.dot(h, self.W2) + self.b2
        # Identity or projection shortcut
        if self.use_projection:
            shortcut = np.dot(x, self.Ws) + self.bs
        else:
            shortcut = x
        out = residual + shortcut
        return self.relu(out)


def residual_block_demo():
    """Demonstrate the residual connection principle."""
    print("=== ResNet 2015 Modern Demonstration ===")
    print("Residual block: y = F(x) + x")
    np.random.seed(42)
    block = ResidualBlockModern(input_dim=10, hidden_dim=10)
    x = np.random.randn(1, 10)
    out = block.forward(x)
    print(f"Input: {x[0][:3]}...")
    print(f"Output: {out[0][:3]}...")
    print("Residual connection demonstrated.")


if __name__ == "__main__":
    residual_block_demo()