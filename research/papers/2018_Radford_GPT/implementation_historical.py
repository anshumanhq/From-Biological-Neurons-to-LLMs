"""
GPT (2018) – Historical Forward-Pass Demonstration
Architecture:
- 12-layer Transformer decoder with causal (masked) self-attention
- Autoregressive language modelling

Note: This is a forward-pass-only demonstration with correct causal masking.
Full training on BooksCorpus is beyond the scope of this archive.
"""

import numpy as np

class CausalMask:
    """Causal mask for autoregressive attention."""
    @staticmethod
    def create(seq_len):
        """Create upper triangular mask (positions cannot attend to future)."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        return mask

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
        """Self-attention with optional causal mask."""
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
        # Self-attention with causal mask + residual
        attn_out = self.self_attn.forward(x, mask)
        x = x + attn_out
        # Feed-forward + residual
        ffn_out = self.relu(np.dot(x, self.W1) + self.b1)
        ffn_out = np.dot(ffn_out, self.W2) + self.b2
        x = x + ffn_out
        return x


class GPT_2018:
    """
    GPT architecture (Radford et al., 2018).
    Decoder-only Transformer with causal masking.
    """
    def __init__(self, d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_len=512):
        self.d_model = d_model
        self.max_len = max_len
        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        print(f"GPT architecture initialized (d_model={d_model}, heads={num_heads}, layers={num_layers}).")
        print("Note: This is a forward-pass-only demonstration with correct causal masking.")

    def forward(self, x):
        """
        Forward pass through the decoder blocks with causal masking.
        x: input sequence (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        mask = CausalMask.create(seq_len)
        for block in self.decoder_blocks:
            x = block.forward(x, mask)
        return x


if __name__ == "__main__":
    print("=== GPT 2018 Forward-Pass Demo ===")
    print("Note: This demonstrates causal masking and decoder-only architecture.")
    np.random.seed(42)
    gpt = GPT_2018(d_model=64, num_heads=4, num_layers=2, d_ff=128)
    # Dummy sequence: batch=1, seq_len=10, d_model=64
    x = np.random.randn(1, 10, 64)
    out = gpt.forward(x)
    print(f"Output shape: {out.shape}")
    print("Forward pass with causal masking complete.")