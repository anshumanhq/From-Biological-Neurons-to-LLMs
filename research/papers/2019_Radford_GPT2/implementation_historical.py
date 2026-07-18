"""
GPT-2 (2019) – Historical Forward-Pass Demonstration
Architecture:
- Decoder-only Transformer with causal masking
- Larger than GPT-1: 1.5B parameters (XL variant)
- Byte-level BPE tokenization (conceptual)

Note: This is a forward-pass-only demonstration with causal masking.
Full training on WebText (40GB) is beyond the scope of this archive.
"""

import numpy as np

class CausalMask:
    """Causal mask for autoregressive attention."""
    @staticmethod
    def create(seq_len):
        """Create upper triangular mask (positions cannot attend to future)."""
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

class ScaledDotProductAttentionWithMask:
    """Scaled dot-product attention with causal masking."""
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        output = np.dot(attn_weights, V)
        return output, attn_weights


class MultiHeadSelfAttentionWithMask:
    """Multi-head self-attention with causal masking."""
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x, mask=None):
        Q = np.dot(x, self.W_Q)
        K = np.dot(x, self.W_K)
        V = np.dot(x, self.W_V)
        heads = []
        for i in range(self.num_heads):
            Q_head = Q[:, i*self.d_k:(i+1)*self.d_k]
            K_head = K[:, i*self.d_k:(i+1)*self.d_k]
            V_head = V[:, i*self.d_k:(i+1)*self.d_k]
            attn = ScaledDotProductAttentionWithMask(self.d_k)
            out, _ = attn.forward(Q_head, K_head, V_head, mask)
            heads.append(out)
        concat = np.concatenate(heads, axis=1)
        output = np.dot(concat, self.W_O)
        return output


class TransformerDecoderBlock:
    """Single Transformer decoder block with causal self-attention."""
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadSelfAttentionWithMask(d_model, num_heads)
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros((1, d_model))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x, mask=None):
        attn_out = self.self_attn.forward(x, mask)
        x = x + attn_out
        ffn_out = self.relu(np.dot(x, self.W1) + self.b1)
        ffn_out = np.dot(ffn_out, self.W2) + self.b2
        x = x + ffn_out
        return x


class GPT2_2019:
    """
    GPT-2 architecture (Radford et al., 2019).
    Larger decoder-only Transformer with causal masking.
    Model sizes: Small (117M), Medium (345M), Large (774M), XL (1.5B).
    """
    def __init__(self, model_size="xl", d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_len=1024):
        self.model_size = model_size
        self.max_len = max_len
        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        print(f"GPT-2 architecture initialized (model_size={model_size}, d_model={d_model}, heads={num_heads}, layers={num_layers}).")
        print("Note: This is a forward-pass-only demonstration with causal masking.")

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        mask = CausalMask.create(seq_len)
        for block in self.decoder_blocks:
            x = block.forward(x, mask)
        return x


if __name__ == "__main__":
    print("=== GPT-2 2019 Forward-Pass Demo ===")
    print("Note: This demonstrates causal masking and decoder-only architecture.")
    print("GPT-2 variants: Small (117M), Medium (345M), Large (774M), XL (1.5B).")
    np.random.seed(42)
    gpt2 = GPT2_2019(model_size="xl", d_model=1600, num_heads=25, num_layers=48, d_ff=6400)
    x = np.random.randn(1, 10, 1600)
    out = gpt2.forward(x)
    print(f"Output shape: {out.shape}")
    print("Forward pass with causal masking complete.")