"""
Hopfield (1982) – Associative Memory Network
NumPy implementation with Hebbian storage and asynchronous energy minimization.
"""

import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    """
    Hopfield network with binary neurons (±1).
    """
    def __init__(self, N):
        self.N = N
        self.weights = np.zeros((N, N))

    def train(self, patterns):
        """
        Store patterns using Hebbian rule.
        patterns: (P, N) array of ±1 values.
        """
        P, N = patterns.shape
        assert N == self.N, "Pattern dimension mismatch."
        self.weights = np.dot(patterns.T, patterns) / N
        np.fill_diagonal(self.weights, 0.0)

    def energy(self, state):
        """Compute the Hopfield energy."""
        return -0.5 * np.dot(state, np.dot(self.weights, state))

    def update_async(self, state, iterations=100, verbose=False):
        """Asynchronous update until convergence."""
        state = state.copy()
        energies = []
        for _ in range(iterations):
            idx = np.random.randint(0, self.N)
            local_field = np.dot(self.weights[idx], state)
            state[idx] = 1 if local_field >= 0 else -1
            energies.append(self.energy(state))
        return state, energies

    def recover(self, corrupted_state, iterations=100):
        """Recover a pattern from a corrupted input."""
        return self.update_async(corrupted_state, iterations)

    def plot_energy(self, energies):
        """Plot the energy over update steps."""
        plt.figure(figsize=(8, 4))
        plt.plot(energies, 'b-', linewidth=2)
        plt.title('Hopfield Network Energy Minimization')
        plt.xlabel('Update Step')
        plt.ylabel('Energy')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    # Define 4 simple 3x3 binary patterns (flattened)
    x1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])  # X
    x2 = np.array([1, 1, 1, 1, -1, 1, 1, 1, 1])    # O
    x3 = np.array([1, -1, 1, 1, -1, 1, -1, 1, -1])  # V
    x4 = np.array([1, -1, 1, 1, 1, 1, 1, -1, 1])    # H

    patterns = np.array([x1, x2, x3, x4])
    N = len(x1)

    hopfield = HopfieldNetwork(N)
    hopfield.train(patterns)

    # Corrupt the 'X' pattern
    corrupted = x1.copy()
    corrupted[[2, 5]] = -corrupted[[2, 5]]  # flip two bits

    print("Original X:\n", x1.reshape(3, 3))
    print("\nCorrupted:\n", corrupted.reshape(3, 3))

    recovered, energies = hopfield.recover(corrupted, iterations=50)
    print("\nRecovered:\n", recovered.reshape(3, 3))

    if np.array_equal(recovered, x1):
        print("\n✅ Success! Recovered the original pattern.")
    else:
        print("\n⚠️ Converged to a spurious state.")

    hopfield.plot_energy(energies)