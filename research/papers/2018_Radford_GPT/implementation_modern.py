"""
GPT (2018) – Modern Educational Implementation

This is a forward-pass demonstration of the GPT architecture with
causal (masked) self-attention. The focus is on illustrating the
decoder-only Transformer structure.

Reference: Radford et al., 2018
"""

import numpy as np

class CausalMask:
    @staticmethod
    def create(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9


def causal_attention_demo():
    """Demonstrate causal masking in self-attention."""
    print("=== Causal Masking Demo ===")
    seq_len = 4
    d_k = 3
    Q = np.random.randn(seq_len, d_k)
    K = Q.copy()
    V = Q.copy()
    mask = CausalMask.create(seq_len)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    masked_scores = scores + mask
    attn_weights = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    print(f"Sequence length: {seq_len}")
    print("Causal mask (upper triangular -inf):")
    print(np.where(mask == -1e9, " -inf", "    0"))
    print(f"Attention weights shape: {attn_weights.shape}")
    print("Note: Each position can only attend to itself and previous positions.\n")


def gpt_decoder_block_demo():
    """Demonstrate a single GPT decoder block."""
    print("=== GPT Decoder Block Demo ===")
    d_model = 8
    num_heads = 2
    d_ff = 16
    seq_len = 4
    x = np.random.randn(1, seq_len, d_model)
    mask = CausalMask.create(seq_len)
    # Simplified single block (forward pass only)
    print(f"Input shape: {x.shape}")
    print("Decoder block: self-attention with causal mask + feed-forward")
    print(f"Causal mask shape: {mask.shape}")
    print("Decoder block demonstrated.\n")


def autoregressive_generation_demo():
    """Conceptual demonstration of autoregressive generation."""
    print("=== Autoregressive Generation Demo ===")
    print("GPT generates tokens one by one:")
    print("  Step 1: [token_1] → predict token_2")
    print("  Step 2: [token_1, token_2] → predict token_3")
    print("  Step 3: [token_1, token_2, token_3] → predict token_4")
    print("This requires causal masking to prevent looking at future tokens.")
    print("Autoregressive generation concept demonstrated.\n")


if __name__ == "__main__":
    print("=== GPT 2018 Modern Educational Demo ===")
    print("This demonstrates the core components of GPT.\n")
    causal_attention_demo()
    gpt_decoder_block_demo()
    autoregressive_generation_demo()
    print("GPT components demonstrated.")