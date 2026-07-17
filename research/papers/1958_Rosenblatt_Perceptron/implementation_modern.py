"""
Rosenblatt (1958) – Perceptron
MODERN ADAPTATION (Pedagogical)

Characteristics:
- Uses the classic Perceptron learning rule
- Demonstrates the concept with NumPy
"""

import numpy as np

class PerceptronModern:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.lr = lr

    def forward(self, x):
        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.forward(xi)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                break

if __name__ == "__main__":
    print("=== Perceptron Modern Demo ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate
    p = PerceptronModern(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)
    preds = [p.forward(xi) for xi in X]
    print(f"Predictions: {preds}")
