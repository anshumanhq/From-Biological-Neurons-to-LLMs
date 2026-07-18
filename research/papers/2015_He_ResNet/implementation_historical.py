"""
ResNet (2015) – Historical Forward-Pass Demonstration
Architecture:
- Residual blocks with identity shortcuts
- Bottleneck blocks for deeper networks
- ReLU activations

Note: This is a forward-pass-only demonstration of the residual block structure.
Full training on ImageNet-scale data is beyond the scope of this archive.
"""

import numpy as np

class ResidualBlock:
    """
    A basic residual block with two convolutional layers and an identity shortcut.
    """
    def __init__(self, input_dim, hidden_dim):
        # Simplified linear layers (convolutions are replaced with linear for demonstration)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b2 = np.zeros((1, input_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Residual mapping: F(x)
        h = self.relu(np.dot(x, self.W1) + self.b1)
        residual = np.dot(h, self.W2) + self.b2
        # Identity shortcut: add x to F(x)
        out = residual + x
        return self.relu(out)


class ResNet_2015:
    """
    ResNet architecture (He et al., 2015).
    Demonstrates the residual block structure.
    """
    def __init__(self, num_blocks=5, input_dim=64):
        self.blocks = [ResidualBlock(input_dim, input_dim) for _ in range(num_blocks)]
        print(f"ResNet architecture initialized with {num_blocks} residual blocks.")
        print("Note: This is a forward-pass-only demonstration.")

    def forward(self, x):
        """Forward pass through the residual blocks."""
        out = x
        for i, block in enumerate(self.blocks):
            out = block.forward(out)
            print(f"Block {i+1} output shape: {out.shape}")
        return out


if __name__ == "__main__":
    print("=== ResNet 2015 Forward-Pass Demo ===")
    print("Note: This demonstrates the residual block structure.")
    np.random.seed(42)
    net = ResNet_2015(num_blocks=3, input_dim=64)
    x = np.random.randn(1, 64)
    out = net.forward(x)
    print(f"Final output shape: {out.shape}")
    print("Forward pass complete.")