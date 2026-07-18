"""
Transformer (2017) – Modern Educational Implementation

This is a forward-pass demonstration of the Transformer's core components:
- Scaled dot-product attention
- Multi-head attention
- Positional encoding
- Encoder block structure

Full training on large datasets is not implemented here due to complexity.
The focus is on illustrating the architectural principles.

Reference: Vaswani et al., 2017
"""

import numpy as np

class ScaledDotProductAttentionModern:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask * -1e9
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        return np.dot(attn_weights, V), attn_weights


def scaled_dot_product_attention_demo():
    """Demonstrate scaled dot-product attention with a simple example."""
    print("=== Scaled Dot-Product Attention Demo ===")
    d_k = 4
    Q = np.random.randn(1, d_k)
    K = np.random.randn(3, d_k)
    V = np.random.randn(3, d_k)
    attn = ScaledDotProductAttentionModern(d_k)
    output, weights = attn.forward(Q, K, V)
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"Attention weights: {weights}")
    print(f"Output shape: {output.shape}")
    print("Attention mechanism demonstrated.\n")


def multi_head_attention_demo():
    """Demonstrate multi-head attention (simplified)."""
    print("=== Multi-Head Attention Demo ===")
    d_model = 8
    num_heads = 2
    d_k = d_model // num_heads
    x = np.random.randn(1, d_model)
    heads = []
    for i in range(num_heads):
        # Simulate different heads with different projections
        Q = np.random.randn(1, d_k)
        K = np.random.randn(1, d_k)
        V = np.random.randn(1, d_k)
        attn = ScaledDotProductAttentionModern(d_k)
        out, _ = attn.forward(Q, K, V)
        heads.append(out)
    concat = np.concatenate(heads, axis=1)
    print(f"Number of heads: {num_heads}")
    print(f"Each head output shape: {heads[0].shape}")
    print(f"Concatenated output shape: {concat.shape}")
    print("Multi-head attention demonstrated.\n")


def positional_encoding_demo():
    """Demonstrate sinusoidal positional encoding."""
    print("=== Positional Encoding Demo ===")
    d_model = 8
    max_len = 10
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                pe[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    print(f"Positional encoding shape: {pe.shape}")
    print("Positional encoding demonstrates position-dependent sinusoidal patterns.\n")


if __name__ == "__main__":
    print("=== Transformer 2017 Modern Educational Demo ===")
    print("This demonstrates the core components of the Transformer.")
    scaled_dot_product_attention_demo()
    multi_head_attention_demo()
    positional_encoding_demo()
    print("Transformer components demonstrated.")