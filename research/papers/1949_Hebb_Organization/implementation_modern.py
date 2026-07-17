"""
Hebb (1949) – Hebbian Learning
MODERN ADAPTATION (Pedagogical)

Characteristics:
- Uses the classic Hebbian rule: Δw = η * pre * post
- Demonstrates the core concept with NumPy
- Note: Hebb's original work was a book without code; this is a modern interpretation.
"""

import numpy as np

def hebbian_update(weights, pre, post, lr=0.01):
    """
    Hebbian learning rule: Δw = η * pre * post
    """
    return weights + lr * np.outer(pre, post)

if __name__ == "__main__":
    print("=== Hebbian Learning Demo ===")
    W = np.array([[0.5, -0.5]])
    pre = np.array([1.0, 0.0])
    post = np.array([1.0])
    W_new = hebbian_update(W, pre, post, lr=0.1)
    print(f"Updated weights: {W_new}")
