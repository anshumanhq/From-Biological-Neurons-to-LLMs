"""
Werbos (1974) – Backpropagation
MODERN ADAPTATION (Pedagogical)

Characteristics:
- Uses backpropagation with sigmoid activation
- Demonstrates the concept with NumPy
- Note: Werbos's original was a PhD thesis; this is a modern educational implementation.
"""

import numpy as np

class MLPBackpropModern:
    def __init__(self, layer_sizes, lr=0.5):
        self.lr = lr
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        current = X
        for W, b in zip(self.weights, self.biases):
            z = np.dot(current, W) + b
            self.zs.append(z)
            current = self.sigmoid(z)
            self.activations.append(current)
        return current

    def backward(self, X, y, output):
        m = X.shape[0]
        d_loss = output - y
        d_z = d_loss * self.sigmoid_derivative(output)
        deltas = [d_z]
        for l in range(len(self.weights) - 1, 0, -1):
            d_a = np.dot(deltas[-1], self.weights[l].T)
            d_z = d_a * self.sigmoid_derivative(self.activations[l])
            deltas.append(d_z)
        deltas = deltas[::-1]
        for l in range(len(self.weights)):
            self.weights[l] -= self.lr * np.dot(self.activations[l].T, deltas[l]) / m
            self.biases[l] -= self.lr * np.sum(deltas[l], axis=0, keepdims=True) / m

    def train(self, X, y, epochs=5000):
        for epoch in range(epochs):
            pred = self.forward(X)
            self.backward(X, y, pred)
            if epoch % 1000 == 0:
                loss = np.mean((pred - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

if __name__ == "__main__":
    print("=== Backpropagation Modern Demo ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR
    net = MLPBackpropModern([2, 2, 1], lr=0.5)
    net.train(X, y, epochs=2000)
    preds = net.forward(X)
    print(f"Predictions: {np.round(preds).flatten()}")
