"""
GPT-3 (2020) – Historical Forward-Pass Demonstration
Architecture:
- Decoder-only Transformer with causal masking
- Scaled to 175B parameters (conceptual demonstration)
- In-context learning: zero-shot, one-shot, few-shot

Note: This is a forward-pass-only demonstration. Full training on the
175B model is beyond the scope of this archive.
"""

import numpy as np

class CausalMask:
    """Causal mask for autoregressive attention."""
    @staticmethod
    def create(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

class ScaledDotProductAttentionWithMask:
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


class GPT3_2020:
    """
    GPT-3 architecture (Brown et al., 2020).
    175B parameters, 96 layers, 12,288 d_model, 96 heads.
    Demonstrates in-context learning: zero-shot, one-shot, few-shot.
    """
    def __init__(self, d_model=12288, num_heads=96, num_layers=96, d_ff=49152):
        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        print(f"GPT-3 architecture conceptual (d_model={d_model}, heads={num_heads}, layers={num_layers}).")
        print("Note: This is a forward-pass-only demonstration.")

    def forward(self, x):
        seq_len = x.shape[1]
        mask = CausalMask.create(seq_len)
        for block in self.decoder_blocks:
            x = block.forward(x, mask)
        return x


def in_context_learning_demo():
    """Demonstrate zero-shot, one-shot, and few-shot prompting."""
    print("=== In-context Learning Demo ===")
    print("Zero-shot: Translate English to French: 'cat' →")
    print("One-shot: Translate English to French: 'dog' → 'chien'")
    print("          Translate English to French: 'cat' →")
    print("Few-shot: Translate English to French: 'dog' → 'chien'")
    print("          Translate English to French: 'bird' → 'oiseau'")
    print("          Translate English to French: 'cat' →")
    print("No gradient updates during evaluation.\n")


if __name__ == "__main__":
    print("=== GPT-3 2020 Forward-Pass Demo ===")
    print("Note: This demonstrates the conceptual architecture and in-context learning.")
    print("GPT-3: 175B parameters, 96 layers, 12,288 d_model, 96 heads.\n")
    np.random.seed(42)
    # Reduced dimensions for demonstration
    gpt3 = GPT3_2020(d_model=128, num_heads=4, num_layers=2, d_ff=512)
    x = np.random.randn(1, 8, 128)
    out = gpt3.forward(x)
    print(f"Output shape: {out.shape}")
    in_context_learning_demo()