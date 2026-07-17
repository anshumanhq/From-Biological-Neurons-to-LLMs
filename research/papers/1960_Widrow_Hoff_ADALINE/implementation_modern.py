"""
Widrow & Hoff (1960) – ADALINE / LMS
MODERN ADAPTATION (Pedagogical)

Characteristics:
- Uses the LMS (Delta) rule
- Demonstrates the concept with NumPy
"""

import numpy as np

class ADALINEModern:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.lr = lr

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            total_error = 0.0
            for xi, target in zip(X, y):
                v = self.forward(xi)
                error = target - v
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += error ** 2
            if total_error < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break

if __name__ == "__main__":
    print("=== ADALINE Modern Demo ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate
    ad = ADALINEModern(input_size=2, lr=0.1)
    ad.train(X, y, epochs=50)
    preds = [ad.forward(xi) for xi in X]
    print(f"Predictions: {preds}")
