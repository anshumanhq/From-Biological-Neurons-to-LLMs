'''
Widrow & Hoff (1960) – ADALINE and LMS Algorithm
Pure NumPy implementation.
'''

import numpy as np

class ADALINE:
    """
    Widrow-Hoff ADALINE (1960).
    """
    def __init__(self, input_size, lr=0.01):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.lr = lr

    def forward_linear(self, x):
        """Return the analog (continuous) output."""
        return np.dot(x, self.weights) + self.bias

    def forward_quantized(self, x):
        """Return the binary output (sign function)."""
        return 1 if self.forward_linear(x) >= 0 else 0

    def train(self, X, y, epochs=10):
        """
        Train using LMS (Delta Rule).
        X: Input matrix (samples x features).
        y: Target continuous values.
        """
        for epoch in range(epochs):
            total_error = 0.0
            for xi, target in zip(X, y):
                v = self.forward_linear(xi)
                error = target - v
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += error ** 2

            if total_error < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break

    def predict_quantized(self, X):
        return np.array([self.forward_quantized(xi) for xi in X])

# Demo
if __name__ == "__main__":
    # AND gate
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])
    ad = ADALINE(input_size=2, lr=0.1)
    ad.train(X, y, epochs=100)
    print(f"Weights: {ad.weights}, Bias: {ad.bias}")
    print(f"Predictions: {ad.predict_quantized(X)}")