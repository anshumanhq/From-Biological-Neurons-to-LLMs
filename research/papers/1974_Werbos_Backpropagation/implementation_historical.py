"""
Werbos (1974) – Backpropagation Implementation
A 2-2-1 XOR network trained from scratch using the chain rule.
"""

import numpy as np

class MLP_Backprop:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, lr=0.5):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        d_loss = output - y
        d_z2 = d_loss * self.sigmoid_derivative(output)
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)

        self.W2 -= self.lr * np.dot(self.a1.T, d_z2) / m
        self.b2 -= self.lr * np.sum(d_z2, axis=0, keepdims=True) / m
        self.W1 -= self.lr * np.dot(X.T, d_z1) / m
        self.b1 -= self.lr * np.sum(d_z1, axis=0, keepdims=True) / m

    def train(self, X, y, epochs=5000):
        for epoch in range(epochs):
            pred = self.forward(X)
            self.backward(X, y, pred)
            if epoch % 1000 == 0:
                loss = np.mean((pred - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

if __name__ == "__main__":
    # XOR – the benchmark problem from Minsky & Papert (1969)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    net = MLP_Backprop(input_size=2, hidden_size=2, output_size=1, lr=0.5)
    net.train(X, y, epochs=5000)

    print("\n=== Final Predictions ===")
    print("Inputs:\n", X)
    print("Predictions (rounded):\n", np.round(net.forward(X)))
    print("Expected:\n", y)
    # Expected output: [0, 1, 1, 0] – XOR solved!