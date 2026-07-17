"""
test_implementations.py — Test NumPy implementations dynamically.
From Biological Neurons to LLMs
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_module_from_path(filepath):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hebb_historical():
    hebb_file = REPO_ROOT / "research" / "papers" / "1949_Hebb_Organization" / "implementation_historical.py"
    if hebb_file.exists():
        return load_module_from_path(str(hebb_file))
    return None


@pytest.fixture(scope="module")
def perceptron_historical():
    p_file = REPO_ROOT / "research" / "papers" / "1958_Rosenblatt_Perceptron" / "implementation_historical.py"
    if p_file.exists():
        return load_module_from_path(str(p_file))
    return None


@pytest.fixture(scope="module")
def adaline_historical():
    ad_file = REPO_ROOT / "research" / "papers" / "1960_Widrow_Hoff_ADALINE" / "implementation_historical.py"
    if ad_file.exists():
        return load_module_from_path(str(ad_file))
    return None


@pytest.fixture(scope="module")
def hopfield_historical():
    hop_file = REPO_ROOT / "research" / "papers" / "1982_Hopfield_Network" / "implementation_historical.py"
    if hop_file.exists():
        return load_module_from_path(str(hop_file))
    return None


@pytest.fixture(scope="module")
def werbos_historical():
    w_file = REPO_ROOT / "research" / "papers" / "1974_Werbos_Backpropagation" / "implementation_historical.py"
    if w_file.exists():
        return load_module_from_path(str(w_file))
    return None


def test_hebbian_update(hebb_historical):
    if hebb_historical is None:
        pytest.skip("Hebb implementation not found")
    hebbian_update = hebb_historical.hebbian_update_historical
    W = np.array([[0.5, -0.5]])
    pre = np.array([1.0])
    post = np.array([1.0, 0.0])
    W_new = hebbian_update(W, pre, post, lr=0.1)
    assert np.allclose(W_new, np.array([[0.6, -0.5]]))


def test_perceptron_xor(perceptron_historical):
    if perceptron_historical is None:
        pytest.skip("Perceptron implementation not found")
    Perceptron = perceptron_historical.PerceptronScratch
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    p = Perceptron(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)
    preds = np.array([p.forward(xi) for xi in X])
    assert not np.array_equal(preds, y)


def test_perceptron_and(perceptron_historical):
    if perceptron_historical is None:
        pytest.skip("Perceptron implementation not found")
    Perceptron = perceptron_historical.PerceptronScratch
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    p = Perceptron(input_size=2, lr=0.1)
    p.train(X, y, epochs=20)
    preds = np.array([p.forward(xi) for xi in X])
    assert np.array_equal(preds, y)


def test_adaline_and(adaline_historical):
    """
    Test ADALINE on AND gate. The implementation now uses bipolar targets
    internally, so we can expect perfect classification with sufficient epochs.
    """
    if adaline_historical is None:
        pytest.skip("ADALINE implementation not found")
    ADALINE = adaline_historical.ADALINE

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # 0/1 labels – converted to bipolar inside train()

    ad = ADALINE(input_size=2, lr=0.1)
    ad.train(X, y, epochs=1000)   # should converge quickly

    preds = ad.predict_quantized(X)
    assert np.array_equal(preds, y)


def test_hopfield_recovery(hopfield_historical):
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
    recovered, _ = hop.recover(corrupted, iterations=50)
    assert np.array_equal(recovered, x1)


def test_xor_backprop(werbos_historical):
    if werbos_historical is None:
        pytest.skip("Backprop implementation not found")
    MLP = werbos_historical.MLP_Backprop
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    net = MLP(input_size=2, hidden_size=4, output_size=1, lr=0.5)
    net.train(X, y, epochs=10000)
    preds = net.forward(X)
    rounded = np.round(preds).flatten()
    expected = y.flatten()
    assert np.mean((rounded == expected)) >= 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
