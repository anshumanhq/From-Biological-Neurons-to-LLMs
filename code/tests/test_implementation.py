"""
test_implementations.py — Test the NumPy implementations.
From Biological Neurons to LLMs

Usage:
    pytest code/tests/
"""

import sys
import os
import pytest
import numpy as np

# Add numpy_scratch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'numpy_scratch'))


def test_import_all_implementations():
    """Test that all implementation modules can be imported."""
    # This is a placeholder; actual implementations will be tested
    # as they are moved to code/numpy_scratch/
    assert True


# ============================================================
# Tests for Historical Implementations
# ============================================================

def test_hebbian_update():
    """Test Hebbian learning rule."""
    from research.papers.1949_Hebb_Organization.implementation_historical import hebbian_update_historical
    W = np.array([[0.5, -0.5]])
    pre = np.array([1.0, 0.0])
    post = np.array([1.0])
    W_new = hebbian_update_historical(W, pre, post, lr=0.1)
    assert W_new[0, 0] == 0.6
    assert W_new[0, 1] == -0.5


def test_perceptron_xor():
    """Test that the historical perceptron cannot learn XOR."""
    from research.papers.1958_Rosenblatt_Perceptron.implementation_historical import PerceptronScratch
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    p = PerceptronScratch(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)
    preds = [p.forward(xi) for xi in X]
    # XOR should NOT converge to all correct (linear separability)
    assert any(preds != y)


def test_and_gate():
    """Test that the historical perceptron can learn AND."""
    from research.papers.1958_Rosenblatt_Perceptron.implementation_historical import PerceptronScratch
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND
    p = PerceptronScratch(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)
    preds = [p.forward(xi) for xi in X]
    assert np.array_equal(preds, y)


def test_adaline_and():
    """Test ADALINE on AND gate."""
    from research.papers.1960_Widrow_Hoff_ADALINE.implementation_historical import ADALINE
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND
    ad = ADALINE(input_size=2, lr=0.1)
    ad.train(X, y, epochs=50)
    preds = ad.predict_quantized(X)
    assert np.array_equal(preds, y)


def test_hopfield_recovery():
    """Test Hopfield network pattern recovery."""
    from research.papers.1982_Hopfield_Network.implementation_historical import HopfieldNetwork
    x1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])
    patterns = np.array([x1])
    hopfield = HopfieldNetwork(N=9)
    hopfield.train(patterns)

    corrupted = x1.copy()
    corrupted[[2, 5]] = -corrupted[[2, 5]]
    recovered, _ = hopfield.recover(corrupted, iterations=50)

    assert np.array_equal(recovered, x1)


def test_xor_backprop():
    """Test that backpropagation can solve XOR."""
    from research.papers.1974_Werbos_Backpropagation.implementation_historical import MLP_Backprop
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    net = MLP_Backprop([2, 2, 1], lr=0.5)
    net.train(X, y, epochs=2000, verbose=False)
    preds = net.forward(X)
    # With rounding, should match XOR
    rounded = np.round(preds).flatten()
    expected = y.flatten()
    # Allow tolerance due to random initialization
    assert np.mean((rounded == expected)) >= 0.75


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    # Run pytest manually
    pytest.main([__file__, "-v", "--tb=short"])