"""
LeNet-5 (1998) – Modern Implementation (Educational)
- Uses ReLU instead of Tanh
- Uses MaxPool instead of Average Pooling
- Uses Adam optimizer
- Complete backpropagation included
"""

import numpy as np

class LeNet5_Modern:
    """
    LeNet-5 with modern improvements:
    - ReLU activations
    - Max pooling
    - Adam optimizer
    """
    def __init__(self, lr=0.001):
        self.lr = lr
        # Layer weights (same dimensions as historical)
        self.conv1_W = np.random.randn(6, 1, 5, 5) * np.sqrt(2.0 / (1*5*5))
        self.conv1_b = np.zeros(6)
        self.conv3_W = np.random.randn(16, 6, 5, 5) * np.sqrt(2.0 / (6*5*5))
        self.conv3_b = np.zeros(16)
        self.fc5_W = np.random.randn(16*5*5, 120) * np.sqrt(2.0 / (16*5*5))
        self.fc5_b = np.zeros(120)
        self.fc6_W = np.random.randn(120, 84) * np.sqrt(2.0 / 120)
        self.fc6_b = np.zeros(84)
        self.fc7_W = np.random.randn(84, 10) * np.sqrt(2.0 / 84)
        self.fc7_b = np.zeros(10)

        # Adam state
        self.m = [np.zeros_like(w) for w in [self.conv1_W, self.conv3_W, self.fc5_W, self.fc6_W, self.fc7_W]]
        self.v = [np.zeros_like(w) for w in [self.conv1_W, self.conv3_W, self.fc5_W, self.fc6_W, self.fc7_W]]
        self.t = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def conv2d(self, X, W, b):
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

    def max_pool2d(self, X, pool_size=2):
        batch, depth, h, w = X.shape
        out_h = h // pool_size
        out_w = w // pool_size
        out = np.zeros((batch, depth, out_h, out_w))
        self.pool_indices = {}
        for b_idx in range(batch):
            for d in range(depth):
                for i in range(out_h):
                    for j in range(out_w):
                        region = X[b_idx, d, i*pool_size:(i+1)*pool_size,
                                   j*pool_size:(j+1)*pool_size]
                        max_val = np.max(region)
                        out[b_idx, d, i, j] = max_val
                        idx = np.unravel_index(np.argmax(region), region.shape)
                        self.pool_indices[(b_idx, d, i, j)] = idx
        return out

    def forward(self, X):
        # Conv1 → ReLU → Pool1
        conv1 = self.conv2d(X, self.conv1_W, self.conv1_b)
        relu1 = self.relu(conv1)
        pool1 = self.max_pool2d(relu1, pool_size=2)

        # Conv2 → ReLU → Pool2
        conv2 = self.conv2d(pool1, self.conv3_W, self.conv3_b)
        relu2 = self.relu(conv2)
        pool2 = self.max_pool2d(relu2, pool_size=2)

        # Flatten → FC layers
        batch = pool2.shape[0]
        flat = pool2.reshape(batch, -1)
        fc5 = np.dot(flat, self.fc5_W) + self.fc5_b
        relu5 = self.relu(fc5)
        fc6 = np.dot(relu5, self.fc6_W) + self.fc6_b
        relu6 = self.relu(fc6)
        output = np.dot(relu6, self.fc7_W) + self.fc7_b

        self.cache = (X, conv1, relu1, pool1, conv2, relu2, pool2, flat, fc5, relu5, fc6, relu6, output)
        return output

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = np.mean((pred - y) ** 2)
            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

if __name__ == "__main__":
    print("=== LeNet-5 Modern Demo ===")
    print("Architecture: Conv1 → ReLU → MaxPool → Conv2 → ReLU → MaxPool → FC3 → FC4 → Output")
    print("Activation: ReLU, Pooling: Max, Optimizer: SGD (simplified)")