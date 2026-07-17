"""
Minsky & Papert (1969) – Impossibility of XOR for a single‑layer Perceptron.
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
        history = []
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.forward(xi)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            history.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                break
        return history

if __name__ == "__main__":
    # XOR – NOT linearly separable
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])

    p = SingleLayerPerceptron(input_size=2, lr=0.1)
    history = p.train(X, y, epochs=100)
    print("Final weights:", p.weights, "bias:", p.bias)
    print("Predictions:", [p.forward(x) for x in X])
    print("Expected:   ", list(y))
    print("\nThe Perceptron cannot converge on XOR – exactly as Minsky & Papert proved.")

    # AND – linearly separable (converges)
    X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])
    p2 = SingleLayerPerceptron(input_size=2, lr=0.1)
    p2.train(X_and, y_and, epochs=20)
    print("\nAND gate predictions:", [p2.forward(x) for x in X_and])
    print("AND is linearly separable; the Perceptron converges.")