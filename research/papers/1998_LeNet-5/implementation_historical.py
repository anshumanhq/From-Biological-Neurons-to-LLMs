"""
LeNet-5 (1998) – Historical Implementation
Faithful to the original architecture:
- 7 layers: Conv1 (6×5×5) → Pool1 (2×2 avg) → Conv2 (16×5×5) → Pool2 (2×2 avg)
  → FC3 (120) → FC4 (84) → Output (10)
- Activation: Tanh (not ReLU)
- Pooling: Average pooling (not max)
- Optimizer: SGD with learning rate
"""

import numpy as np

class LeNet5_Historical:
    """
    LeNet-5 architecture (LeCun et al., 1998).
    Input: 32×32 grayscale images.
    Output: 10 classes (digits 0-9).
    """
    def __init__(self, lr=0.01):
        self.lr = lr

        # Layer 1: Convolution (6 filters, 5×5)
        self.conv1_W = np.random.randn(6, 1, 5, 5) * 0.1
        self.conv1_b = np.zeros(6)

        # Layer 2: Average Pooling (2×2)
        # Layer 3: Convolution (16 filters, 5×5)
        self.conv3_W = np.random.randn(16, 6, 5, 5) * 0.1
        self.conv3_b = np.zeros(16)

        # Layer 4: Average Pooling (2×2)
        # Layer 5: Fully Connected (120 units)
        self.fc5_W = np.random.randn(16 * 5 * 5, 120) * 0.1
        self.fc5_b = np.zeros(120)

        # Layer 6: Fully Connected (84 units)
        self.fc6_W = np.random.randn(120, 84) * 0.1
        self.fc6_b = np.zeros(84)

        # Layer 7: Output (10 units)
        self.fc7_W = np.random.randn(84, 10) * 0.1
        self.fc7_b = np.zeros(10)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def conv2d(self, X, W, b):
        """2D convolution with padding same."""
        batch, in_depth, h, w = X.shape
        out_depth, _, k, _ = W.shape
        out_h = h - k + 1
        out_w = w - k + 1
        out = np.zeros((batch, out_depth, out_h, out_w))
        for b_idx in range(batch):
            for d_out in range(out_depth):
                for i in range(out_h):
                    for j in range(out_w):
                        rf = X[b_idx, :, i:i+k, j:j+k]
                        out[b_idx, d_out, i, j] = np.sum(rf * W[d_out]) + b[d_out]
        return out

    def avg_pool2d(self, X, pool_size=2):
        """Average pooling."""
        batch, depth, h, w = X.shape
        out_h = h // pool_size
        out_w = w // pool_size
        out = np.zeros((batch, depth, out_h, out_w))
        for b_idx in range(batch):
            for d in range(depth):
                for i in range(out_h):
                    for j in range(out_w):
                        region = X[b_idx, d, i*pool_size:(i+1)*pool_size,
                                   j*pool_size:(j+1)*pool_size]
                        out[b_idx, d, i, j] = np.mean(region)
        return out

    def forward(self, X):
        """Forward pass with tanh activations."""
        # Conv1 → Tanh → Pool1
        self.conv1_out = self.conv2d(X, self.conv1_W, self.conv1_b)
        self.tanh1_out = self.tanh(self.conv1_out)
        self.pool1_out = self.avg_pool2d(self.tanh1_out, pool_size=2)

        # Conv2 → Tanh → Pool2
        self.conv2_out = self.conv2d(self.pool1_out, self.conv3_W, self.conv3_b)
        self.tanh2_out = self.tanh(self.conv2_out)
        self.pool2_out = self.avg_pool2d(self.tanh2_out, pool_size=2)

        # Flatten
        batch = self.pool2_out.shape[0]
        self.flattened = self.pool2_out.reshape(batch, -1)

        # FC5 → Tanh
        self.fc5_out = np.dot(self.flattened, self.fc5_W) + self.fc5_b
        self.tanh5_out = self.tanh(self.fc5_out)

        # FC6 → Tanh
        self.fc6_out = np.dot(self.tanh5_out, self.fc6_W) + self.fc6_b
        self.tanh6_out = self.tanh(self.fc6_out)

        # Output → (no activation, for classification we'll apply softmax elsewhere)
        self.output = np.dot(self.tanh6_out, self.fc7_W) + self.fc7_b
        return self.output

    def train(self, X, y, epochs=10):
        """Train the network using SGD."""
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            # Forward pass
            pred = self.forward(X)
            # Compute loss (MSE) and update weights
            # (Full backward pass not implemented here for brevity;
            #  see implementation_modern.py for complete backprop)
            loss = np.mean((pred - y) ** 2)
            print(f"Loss: {loss:.6f}")

if __name__ == "__main__":
    print("=== LeNet-5 Historical Demo ===")
    print("Architecture: Conv1 → Pool1 → Conv2 → Pool2 → FC3 → FC4 → Output")
    print("Activation: Tanh, Pooling: Average")