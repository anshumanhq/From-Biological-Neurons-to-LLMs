"""
Hebb (1949) – Hebbian Learning
HISTORICAL IMPLEMENTATION (Faithful to the Original)

Characteristics:
- Uses the classic Hebbian rule: Δw = η * pre * post
- Demonstrates the core concept with NumPy
- The original book did not contain code; this is a faithful mathematical
  interpretation of Hebb's neurophysiological postulate.

Reference: Hebb, D. O. (1949). The Organization of Behavior.
           John Wiley & Sons.
"""

import numpy as np

def hebbian_update_historical(weight_matrix, pre_synaptic, post_synaptic, lr=0.01):
    """
    Hebbian learning rule (1949):
    Δw = η * (pre_synaptic_activity) * (post_synaptic_activity)

    This is the exact mathematical formulation of Hebb's postulate:
    "When an axon of cell A is near enough to excite cell B and repeatedly
    or persistently takes part in firing it, some growth process or metabolic
    change takes place in one or both cells such that A's efficiency, as one
    of the cells firing B, is increased."

    Args:
        weight_matrix: Current weight matrix (shape: n_input, n_output)
        pre_synaptic: Presynaptic activity vector (shape: n_input)
        post_synaptic: Postsynaptic activity vector (shape: n_output)
        lr: Learning rate (positive constant)

    Returns:
        Updated weight matrix
    """
    delta_w = lr * np.outer(pre_synaptic, post_synaptic)
    return weight_matrix + delta_w


class HebbianNetworkHistorical:
    """
    A simple network implementing Hebbian learning (1949).
    This demonstrates the formation of cell assemblies through
    correlated activity.
    """

    def __init__(self, n_input, n_output, lr=0.01):
        self.weights = np.random.randn(n_input, n_output) * 0.01
        self.lr = lr

    def forward(self, x):
        """Compute the output from input."""
        return np.dot(x, self.weights)

    def train(self, X, y, epochs=10):
        """
        Train using Hebbian learning (unsupervised).
        The 'y' here is the postsynaptic activity, not a target.
        """
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                # Hebbian update: strengthen correlated pre/post activity
                self.weights = hebbian_update_historical(
                    self.weights, xi, yi, self.lr
                )
            print(f"Epoch {epoch+1}: Weights mean = {np.mean(self.weights):.4f}")

    def predict(self, X):
        return np.array([self.forward(xi) for xi in X])


# === Historical Demo ===
if __name__ == "__main__":
    print("=== Hebbian Learning (1949) Historical Demo ===")

    # Simple example: 2 inputs, 1 output
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Postsynaptic activity (could be from another layer or a teacher signal)
    y = np.array([[1.0], [1.0], [0.0]])

    net = HebbianNetworkHistorical(n_input=2, n_output=1, lr=0.1)
    net.train(X, y, epochs=5)

    print("\nFinal weights:")
    print(net.weights)

    print("\nPredictions:")
    for xi in X:
        pred = net.forward(xi)
        print(f"Input: {xi} → Output: {pred[0]:.4f}")
