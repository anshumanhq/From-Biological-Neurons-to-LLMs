"""
Rumelhart, Hinton & Williams (1986) – Backpropagation
HISTORICAL IMPLEMENTATION (Faithful to the Original Work)

Characteristics:
- Weight initialization: small uniform random values (typically in [-0.3, 0.3]),
  consistent with the practices described in the original 1986 paper.
- Update rule: online (stochastic) gradient descent, updating weights after
  every single training example, as was commonly used in the original experiments.
- No momentum, no adaptive learning rates.

This implementation prioritises historical accuracy over modern performance.
"""

import numpy as np

class MLP_Backprop_1986_Historical:
    """
    Multi-layer Perceptron with backpropagation.
    Designed to replicate the experimental setup described in the 1986 Nature paper.
    """
    def __init__(self, layer_sizes, lr=0.5, weight_range=0.3):
        """
        layer_sizes: List of integers, e.g., [2, 2, 1] for XOR.
        lr: Learning rate (typically 0.5 in the original experiments).
        weight_range: Uniform random weights in [-weight_range, weight_range].
        """
        self.lr = lr
        self.weights = []
        self.biases = []

        # Historical initialization: small uniform random values.
        # The original paper typically used values in the range [-0.3, 0.3].
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            w = np.random.uniform(-weight_range, weight_range, size=(fan_in, fan_out))
            b = np.random.uniform(-weight_range, weight_range, size=(1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        """Forward pass for a single sample or mini-batch."""
        self.activations = [X]
        self.zs = []
        current = X
        for W, b in zip(self.weights, self.biases):
            z = np.dot(current, W) + b
            self.zs.append(z)
            current = self.sigmoid(z)
            self.activations.append(current)
        return current

    def backward_online(self, x, y):
        """
        Backward pass for a SINGLE sample (online / stochastic update).
        This matches the update strategy commonly used in the original 1986 work.
        """
        # Forward pass for a single sample
        output = self.forward(x.reshape(1, -1))
        y = np.array(y).reshape(1, -1)

        # Output layer delta
        d_loss = output - y
        d_z = d_loss * self.sigmoid_derivative(output)
        deltas = [d_z]

        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 1, 0, -1):
            d_a = np.dot(deltas[-1], self.weights[l].T)
            d_z = d_a * self.sigmoid_derivative(self.activations[l])
            deltas.append(d_z)

        deltas = deltas[::-1]

        # Online update: adjust weights immediately for this sample
        for l in range(len(self.weights)):
            # Note: No division by batch size (m=1) because this is online SGD.
            self.weights[l] -= self.lr * np.dot(self.activations[l].T, deltas[l])
            self.biases[l] -= self.lr * np.sum(deltas[l], axis=0, keepdims=True)

        return output

    def train_online(self, X, y, epochs=5000, verbose=True):
        """
        Train using online (stochastic) gradient descent.
        The order of samples is shuffled each epoch for stochasticity.
        """
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the training data each epoch (standard practice)
            indices = np.random.permutation(n_samples)
            for idx in indices:
                self.backward_online(X[idx], y[idx])

            if verbose and epoch % 1000 == 0:
                pred = self.forward(X)
                loss = np.mean((pred - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

# === XOR Demo (Historical Implementation) ===
if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    net_hist = MLP_Backprop_1986_Historical([2, 2, 1], lr=0.5, weight_range=0.3)
    net_hist.train_online(X, y, epochs=5000, verbose=True)

    print("\n=== Historical Implementation: Final Predictions ===")
    preds = net_hist.predict(X)
    print("Inputs:\n", X)
    print("Predictions (rounded):\n", np.round(preds))
    print("Expected:\n", y)