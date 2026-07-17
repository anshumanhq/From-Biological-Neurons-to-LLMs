'''
Rosenblatt (1958) – Perceptron Learning Algorithm
Pure NumPy implementation, no external libraries (except for testing).
'''

import numpy as np

class PerceptronScratch:
    """
    Rosenblatt's Perceptron (1958).
    """
    def __init__(self, input_size, lr=0.1, threshold=0.0):
        # He initialization (small random weights)
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = -threshold  # threshold to bias conversion
        self.lr = lr

    def forward(self, x):
        """Binary step activation."""
        linear_out = np.dot(x, self.weights) + self.bias
        return 1 if linear_out > 0 else 0

    def train(self, X, y, epochs=10):
        """Training loop using the Perceptron convergence rule."""
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.forward(xi)
                error = target - pred
                if error != 0:
                    # Update weights and bias
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            # Early stopping if all samples are correctly classified
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                break

    def predict(self, X):
        return np.array([self.forward(xi) for xi in X])

# Test: AND gate (linear separable)
if __name__ == "__main__":
    print("=== Test: AND gate ===")
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])  # AND labels

    p = PerceptronScratch(input_size=2, lr=0.1)
    p.train(X_train, y_train, epochs=20)
    preds = p.predict(X_train)
    print(f"Predictions: {preds}")
    print(f"Expected:    {y_train}")

    # Test: XOR gate (not linearly separable - will NOT converge)
    print("\n=== Test: XOR gate (should NOT converge) ===")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR labels
    p_xor = PerceptronScratch(input_size=2, lr=0.1)
    p_xor.train(X_xor, y_xor, epochs=10)
    preds_xor = p_xor.predict(X_xor)
    print(f"Predictions: {preds_xor}")
    print(f"Expected:    {y_xor}")
    print("Notice: The Perceptron cannot learn XOR - it oscillates.")