"""
Seq2Seq (2014) – Historical Forward-Pass Demonstration
Architecture:
- Encoder LSTM: processes source sequence into a context vector
- Decoder LSTM: generates target sequence from the context vector

Note: This is a forward-pass-only demonstration of the architecture.
Full training with backpropagation through time (BPTT) is not implemented
here due to its complexity.
"""

import numpy as np

class LSTMCell:
    """Simplified LSTM cell for demonstration."""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.W_i = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_c = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_o = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.b_i = np.zeros((1, hidden_size))
        self.b_c = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        combined = np.concatenate([h_prev, x], axis=1)
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        c_tilde = self.tanh(np.dot(combined, self.W_c) + self.b_c)
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        c_next = c_prev + i * c_tilde
        h_next = o * self.tanh(c_next)
        return h_next, c_next


class Seq2Seq_2014:
    """
    Seq2Seq architecture (Sutskever et al., 2014).
    Encoder: processes source sequence into context vector.
    Decoder: generates target sequence from context vector.
    """
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, embedding_size):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # Encoder and decoder LSTMs
        self.encoder_cell = LSTMCell(embedding_size, hidden_size)
        self.decoder_cell = LSTMCell(embedding_size, hidden_size)

        # Embedding matrices (simplified)
        self.encoder_embedding = np.random.randn(input_vocab_size, embedding_size) * 0.1
        self.decoder_embedding = np.random.randn(output_vocab_size, embedding_size) * 0.1

        # Output projection (decoder hidden -> output vocab)
        self.W_out = np.random.randn(hidden_size, output_vocab_size) * 0.1
        self.b_out = np.zeros((1, output_vocab_size))

        print("Seq2Seq architecture initialized (forward-pass only).")

    def embed(self, indices, embedding_matrix):
        """Convert token indices to embeddings."""
        return embedding_matrix[indices]

    def encode(self, source_tokens):
        """
        Encode the source sequence.
        source_tokens: list of token indices (length T)
        Returns: final hidden state (context vector)
        """
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))

        for token_idx in source_tokens:
            x = self.embed(token_idx, self.encoder_embedding).reshape(1, -1)
            h, c = self.encoder_cell.forward(x, h, c)

        return h, c

    def decode(self, context_h, context_c, target_tokens, max_length=10):
        """
        Decode the target sequence from the context vector.
        target_tokens: list of token indices (for teacher forcing)
        Returns: sequence of output token probabilities (logits)
        """
        h = context_h.copy()
        c = context_c.copy()

        outputs = []
        # Start token (simplified: use a dummy token)
        current_token = 0  # START token index (placeholder)

        for t in range(max_length):
            # Embed the current token
            x = self.embed(current_token, self.decoder_embedding).reshape(1, -1)
            h, c = self.decoder_cell.forward(x, h, c)

            # Output projection (logits)
            logits = np.dot(h, self.W_out) + self.b_out
            outputs.append(logits)

            # Teacher forcing: use target token at training time
            if t < len(target_tokens):
                current_token = target_tokens[t]
            else:
                # Greedy decoding (simplified)
                current_token = np.argmax(logits)

        return np.array(outputs)

    def forward(self, source_tokens, target_tokens=None, max_length=10):
        """
        Full forward pass: encode then decode.
        """
        context_h, context_c = self.encode(source_tokens)
        outputs = self.decode(context_h, context_c, target_tokens or [], max_length)
        return outputs


if __name__ == "__main__":
    print("=== Seq2Seq 2014 Forward-Pass Demo ===")
    print("Note: This is a conceptual demonstration. Full training requires BPTT.")
    seq2seq = Seq2Seq_2014(
        input_vocab_size=100,
        output_vocab_size=100,
        hidden_size=128,
        embedding_size=64
    )

    # Example: translate a simple sequence
    source = [5, 12, 7]          # token indices
    target = [3, 8, 15, 2]       # token indices

    outputs = seq2seq.forward(source, target, max_length=5)
    print(f"Output logits shape: {outputs.shape}")
    print("Forward pass complete.")