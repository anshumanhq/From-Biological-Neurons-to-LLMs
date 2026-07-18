"""
LLaMA (2023) – Modern Educational Implementation

This is a conceptual demonstration of LLaMA's efficiency improvements.

Reference: Touvron et al., 2023
"""

import numpy as np

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.dim = dim
        self.eps = eps
        self.g = np.ones(dim)

    def forward(self, x):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * x / rms


def swish(x):
    return x / (1 + np.exp(-np.clip(x, -250, 250)))


def swiglu_demo():
    print("=== SwiGLU Activation Demo ===")
    x = np.random.randn(1, 4)
    W_g = np.random.randn(4, 8) * 0.1
    W_u = np.random.randn(4, 8) * 0.1
    b_g = np.zeros((1, 8))
    b_u = np.zeros((1, 8))
    W_d = np.random.randn(8, 4) * 0.1
    b_d = np.zeros((1, 4))
    gate = swish(np.dot(x, W_g) + b_g)
    value = np.dot(x, W_u) + b_u
    out = np.dot(gate * value, W_d) + b_d
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("SwiGLU gated feed-forward demonstrated.\n")


if __name__ == "__main__":
    print("=== LLaMA 2023 Modern Demo ===")
    swiglu_demo()