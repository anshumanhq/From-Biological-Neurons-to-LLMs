"""
Hopfield (1982) – Hopfield Network
MODERN ADAPTATION (Pedagogical)

Characteristics:
- Uses energy-based recurrent network
- Demonstrates associative memory with NumPy
"""

import numpy as np

class HopfieldNetworkModern:
    def __init__(self, N):
        self.N = N
        self.weights = np.zeros((N, N))

    def train(self, patterns):
        P, N = patterns.shape
        assert N == self.N
        self.weights = np.dot(patterns.T, patterns) / N
        np.fill_diagonal(self.weights, 0.0)

    def energy(self, state):
        return -0.5 * np.dot(state, np.dot(self.weights, state))

    def update_async(self, state, iterations=100):
        state = state.copy()
        for _ in range(iterations):
            idx = np.random.randint(0, self.N)
            local_field = np.dot(self.weights[idx], state)
            state[idx] = 1 if local_field >= 0 else -1
        return state

    def recover(self, corrupted_state, iterations=100):
        return self.update_async(corrupted_state, iterations)

if __name__ == "__main__":
    print("=== Hopfield Network Modern Demo ===")
    np.random.seed(42)
    # Simple 3x3 patterns
    x1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])  # X
    patterns = np.array([x1])
    hopfield = HopfieldNetworkModern(N=9)
    hopfield.train(patterns)
    corrupted = x1.copy()
    corrupted[[2, 5]] = -corrupted[[2, 5]]
    recovered = hopfield.recover(corrupted)
    print("Recovered pattern:", recovered.reshape(3, 3))
