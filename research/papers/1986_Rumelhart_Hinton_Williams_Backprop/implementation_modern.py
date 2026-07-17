"""
Rumelhart, Hinton & Williams (1986) – Backpropagation
MODERN IMPLEMENTATION (Pedagogical / Educational)

Characteristics:
- Weight initialization: He or Xavier initialisation (modern best practices).
- Update rule: Mini-batch gradient descent.
- Optimiser: Adam (or SGD with momentum) – clearly labelled as modern additions.

This implementation is provided for educational comparison. It is NOT what
the 1986 paper used, but it illustrates how the algorithm is implemented today.
"""

import numpy as np

class MLP_Backprop_Modern:
    """
    Modern multi-layer Perceptron with backpropagation and best practices.
    """
    def __init__(self, layer_sizes, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        layer_sizes: List of integers, e.g., [2, 2, 1] for XOR.
        lr: Learning rate (Adam default is typically 0.001).
        beta1, beta2, epsilon: Adam optimiser parameters.
        """
        self.lr = lr
        self.weights = []
        self.biases = []
        self.activations = []
        self.zs = []

        # MODERN INITIALIZATION (He initialization for ReLU; here we use it for sigmoid too)
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            # He initialization (commonly used for ReLU; close enough for sigmoid educational purposes)
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

        # Adam optimiser state (momentum and velocity)
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

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
        self.t += 1

        # Output layer delta
        d_loss = output - y
        d_z = d_loss * self.sigmoid_derivative(output)
        deltas = [d_z]

        for l in range(len(self.weights) - 1, 0, -1):
            d_a = np.dot(deltas[-1], self.weights[l].T)
            d_z = d_a * self.sigmoid_derivative(self.activations[l])
            deltas.append(d_z)

        deltas = deltas[::-1]

        # Compute gradients (averaged over mini-batch)
        grad_w = [np.dot(self.activations[l].T, deltas[l]) / m for l in range(len(self.weights))]
        grad_b = [np.sum(deltas[l], axis=0, keepdims=True) / m for l in range(len(self.weights))]

        # Adam update (modern)
        for l in range(len(self.weights)):
            self.m_w[l] = self.beta1 * self.m_w[l] + (1 - self.beta1) * grad_w[l]
            self.v_w[l] = self.beta2 * self.v_w[l] + (1 - self.beta2) * (grad_w[l] ** 2)
            m_w_hat = self.m_w[l] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[l] / (1 - self.beta2 ** self.t)
            self.weights[l] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * grad_b[l]
            self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (grad_b[l] ** 2)
            m_b_hat = self.m_b[l] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[l] / (1 - self.beta2 ** self.t)
            self.biases[l] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def train(self, X, y, epochs=5000, verbose=True):
        for epoch in range(epochs):
            pred = self.forward(X)
            self.backward(X, y, pred)
            if verbose and epoch % 1000 == 0:
                loss = np.mean((pred - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

# === XOR Demo (Modern Implementation) ===
if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    net_modern = MLP_Backprop_Modern([2, 2, 1], lr=0.01)
    print("=== Modern Implementation (Adam + He init) ===")
    net_modern.train(X, y, epochs=5000, verbose=True)

    print("\n=== Final Predictions ===")
    preds = net_modern.predict(X)
    print("Inputs:\n", X)
    print("Predictions (rounded):\n", np.round(preds))
    print("Expected:\n", y)