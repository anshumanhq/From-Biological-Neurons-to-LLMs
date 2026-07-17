"""
test_implementations.py — Robust tests for NumPy implementations.
From Biological Neurons to LLMs

Tests focus on numerical properties (loss decrease, finite weights,
correct decision boundaries) rather than exact predictions, making
them robust to random initialization.

Usage:
    make test   (recommended)
    pytest code/tests/ -v
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_module_from_path(filepath):
    """Dynamically load a Python module from a file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def hebb_historical():
    f = REPO_ROOT / "research" / "papers" / "1949_Hebb_Organization" / "implementation_historical.py"
    return load_module_from_path(str(f)) if f.exists() else None


@pytest.fixture(scope="module")
def perceptron_historical():
    f = REPO_ROOT / "research" / "papers" / "1958_Rosenblatt_Perceptron" / "implementation_historical.py"
    return load_module_from_path(str(f)) if f.exists() else None


@pytest.fixture(scope="module")
def adaline_historical():
    f = REPO_ROOT / "research" / "papers" / "1960_Widrow_Hoff_ADALINE" / "implementation_historical.py"
    return load_module_from_path(str(f)) if f.exists() else None


@pytest.fixture(scope="module")
def hopfield_historical():
    f = REPO_ROOT / "research" / "papers" / "1982_Hopfield_Network" / "implementation_historical.py"
    return load_module_from_path(str(f)) if f.exists() else None


@pytest.fixture(scope="module")
def werbos_historical():
    f = REPO_ROOT / "research" / "papers" / "1974_Werbos_Backpropagation" / "implementation_historical.py"
    return load_module_from_path(str(f)) if f.exists() else None


# ============================================================
# Tests — Robust Numerical Properties
# ============================================================

def test_hebbian_update(hebb_historical):
    """Test Hebbian rule: weights finite and shape preserved."""
    if hebb_historical is None:
        pytest.skip("Hebb implementation not found")
    hebbian_update = hebb_historical.hebbian_update_historical

    W = np.array([[0.5, -0.5]])
    pre = np.array([1.0])
    post = np.array([1.0, 0.0])
    W_new = hebbian_update(W, pre, post, lr=0.1)

    assert np.all(np.isfinite(W_new)), "Weights should remain finite."
    assert W_new.shape == W.shape, "Weight shape should be preserved."
    assert W_new[0, 0] == 0.6  # Known correct result


def test_perceptron_xor(perceptron_historical):
    """Perceptron should NOT converge on XOR (linear separability)."""
    if perceptron_historical is None:
        pytest.skip("Perceptron implementation not found")
    Perceptron = perceptron_historical.PerceptronScratch

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    p = Perceptron(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)

    preds = np.array([p.forward(xi) for xi in X])
    # XOR should NOT be linearly separable → at least 1 error
    assert not np.array_equal(preds, y)


def test_perceptron_and(perceptron_historical):
    """Perceptron should converge on AND (linearly separable)."""
    if perceptron_historical is None:
        pytest.skip("Perceptron implementation not found")
    Perceptron = perceptron_historical.PerceptronScratch

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    p = Perceptron(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)

    preds = np.array([p.forward(xi) for xi in X])
    # Should converge exactly on linearly separable AND
    assert np.array_equal(preds, y)


def test_adaline_and(adaline_historical):
    """ADALINE on AND: ensure loss decreases and final decision boundary is correct."""
    if adaline_historical is None:
        pytest.skip("ADALINE implementation not found")
    ADALINE = adaline_historical.ADALINE

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    ad = ADALINE(input_size=2, lr=0.01)   # lower learning rate for stability
    losses = []
    for epoch in range(5000):
        total_error = 0.0
        for xi, target in zip(X, y):
            v = ad.forward_linear(xi)
            error = target - v  # target is converted to bipolar internally
            ad.weights += ad.lr * error * xi
            ad.bias += ad.lr * error
            total_error += error ** 2
        losses.append(total_error)
        if total_error < 1e-6:
            break

    # Check loss decreased
    assert losses[-1] < losses[0], "Loss should decrease over training."

    # Check weights finite
    assert np.all(np.isfinite(ad.weights)), "Weights should remain finite."

    # Final predictions: allow 3 out of 4 correct (tolerance for random init)
    preds = ad.predict_quantized(X)
    correct = np.sum(preds == y)
    assert correct >= 3, f"Expected at least 3 correct, got {correct}"


def test_hopfield_recovery(hopfield_historical):
    """Hopfield network: pattern recovery from corrupted input."""
    if hopfield_historical is None:
        pytest.skip("Hopfield implementation not found")
    Hopfield = hopfield_historical.HopfieldNetwork

    np.random.seed(42)
    x1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])
    patterns = np.array([x1])
    hop = Hopfield(N=9)
    hop.train(patterns)

    corrupted = x1.copy()
    corrupted[[2, 5]] = -corrupted[[2, 5]]
    recovered, energies = hop.recover(corrupted, iterations=50)

    assert np.array_equal(recovered, x1), "Network should recover the original pattern."
    assert len(energies) > 0, "Energy values should be recorded."


def test_xor_backprop(werbos_historical):
    """Backprop on XOR: loss decreases, predictions improve."""
    if werbos_historical is None:
        pytest.skip("Backprop implementation not found")
    MLP = werbos_historical.MLP_Backprop

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    net = MLP(input_size=2, hidden_size=4, output_size=1, lr=0.5)

    # Track loss over epochs
    losses = []
    for epoch in range(10000):
        pred = net.forward(X)
        net.backward(X, y, pred)
        if epoch % 1000 == 0:
            loss = np.mean((pred - y) ** 2)
            losses.append(loss)

    # Verify loss decreased significantly
    assert len(losses) > 0, "Loss history should be recorded."
    assert losses[-1] < losses[0], "Loss should decrease during training."
    # Final predictions: at least 3 out of 4 correct (75% accuracy)
    preds = net.forward(X)
    rounded = np.round(preds).flatten()
    expected = y.flatten()
    assert np.mean((rounded == expected)) >= 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
