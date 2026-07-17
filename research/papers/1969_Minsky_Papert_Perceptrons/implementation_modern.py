"""
Minsky & Papert (1969) – XOR Impossibility Demo
MODERN ADAPTATION (Pedagogical)

This demonstrates the impossibility proof:
A single-layer Perceptron cannot learn XOR.
"""

import numpy as np

class SingleLayerPerceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.lr = lr

    def forward(self, x):
        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0

    def train(self, X, y, epochs=100):
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
                return
        print("Did not converge — XOR is not linearly separable.")

if __name__ == "__main__":
    print("=== XOR Impossibility Demo ===")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    p = SingleLayerPerceptron(input_size=2, lr=0.1)
    p.train(X_xor, y_xor, epochs=50)
    preds = [p.forward(xi) for xi in X_xor]
    print(f"Predictions: {preds}")
    print("Expected:   [0, 1, 1, 0]")
