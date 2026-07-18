"""
Transformer (2017) – Historical Forward-Pass Demonstration
Architecture:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Encoder and Decoder block structure

Note: This is a forward-pass-only demonstration of the attention mechanism.
Full training on WMT 2014 is beyond the scope of this archive.
"""

import numpy as np

class ScaledDotProductAttention:
    """Scaled dot-product attention: softmax(QK^T/√d_k)V"""
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask * -1e9
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        output = np.dot(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention:
    """Multi-head attention: Concat(heads) * W_O"""
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, Q, K, V, mask=None):
        # Project Q, K, V
        Q_proj = np.dot(Q, self.W_Q)
        K_proj = np.dot(K, self.W_K)
        V_proj = np.dot(V, self.W_V)
        # Split into heads (simplified: no batch dimension)
        heads = []
        for i in range(self.num_heads):
            Q_head = Q_proj[:, i*self.d_k:(i+1)*self.d_k]
            K_head = K_proj[:, i*self.d_k:(i+1)*self.d_k]
            V_head = V_proj[:, i*self.d_k:(i+1)*self.d_k]
            attn = ScaledDotProductAttention(self.d_k)
            out, _ = attn.forward(Q_head, K_head, V_head, mask)
            heads.append(out)
        # Concatenate heads
        concat = np.concatenate(heads, axis=1)
        output = np.dot(concat, self.W_O)
        return output


class PositionalEncoding:
    """Sin/cos positional encoding for sequences."""
    def __init__(self, d_model, max_len=100):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (2*i/d_model)))
                if i+1 < d_model:
                    pe[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
        self.pe = pe

    def forward(self, x, pos):
        return x + self.pe[pos:pos+x.shape[0], :]


class TransformerEncoderBlock:
    """Single encoder block: Self-Attention + FFN with residual connections."""
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros((1, d_model))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Self-Attention + Residual + LayerNorm (simplified)
        attn_out = self.self_attn.forward(x, x, x)
        x = x + attn_out
        # FFN + Residual
        ffn_out = self.relu(np.dot(x, self.W1) + self.b1)
        ffn_out = np.dot(ffn_out, self.W2) + self.b2
        x = x + ffn_out
        return x


class Transformer_2017:
    """
    Transformer architecture (Vaswani et al., 2017).
    Demonstrates the attention-based encoder-decoder structure.
    """
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=100):
        self.d_model = d_model
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_blocks = [TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        print(f"Transformer architecture initialized (d_model={d_model}, heads={num_heads}, layers={num_layers}).")
        print("Note: This is a forward-pass-only demonstration.")

    def encode(self, x):
        """Encoder: add positional encoding, pass through encoder blocks."""
        x = self.pos_enc.forward(x, 0)
        for block in self.encoder_blocks:
            x = block.forward(x)
        return x

    def forward(self, x):
        """Full forward pass: encode (simplified, no decoder)."""
        encoded = self.encode(x)
        return encoded


if __name__ == "__main__":
    print("=== Transformer 2017 Forward-Pass Demo ===")
    print("Note: This demonstrates the attention mechanism and encoder structure.")
    np.random.seed(42)
    transformer = Transformer_2017(d_model=64, num_heads=4, num_layers=2, d_ff=128)
    # Dummy sequence: batch=1, seq_len=10, d_model=64
    x = np.random.randn(1, 10, 64)
    out = transformer.forward(x)
    print(f"Output shape: {out.shape}")
    print("Forward pass complete.")