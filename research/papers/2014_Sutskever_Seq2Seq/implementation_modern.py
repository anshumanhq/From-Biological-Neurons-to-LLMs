"""
Seq2Seq (2014) – Modern Educational Implementation
Architecture:
- Encoder LSTM: processes input sequence.
- Decoder LSTM: generates output sequence with teacher forcing.

This is a simplified educational implementation using NumPy.
"""

import numpy as np

class LSTMCellModern:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.W_i = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_f = np.random.randn(hidden_size + input_size, hidden_size) * 0.1  # modern forget gate
        self.W_c = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_o = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.b_i = np.zeros((1, hidden_size))
        self.b_f = np.zeros((1, hidden_size))
        self.b_c = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        combined = np.concatenate([h_prev, x], axis=1)
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        f = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
        c_tilde = self.tanh(np.dot(combined, self.W_c) + self.b_c)
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        c_next = f * c_prev + i * c_tilde
        h_next = o * self.tanh(c_next)
        return h_next, c_next


class Seq2SeqModern:
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        self.hidden_size = hidden_size
        self.encoder_cell = LSTMCellModern(input_vocab_size, hidden_size)
        self.decoder_cell = LSTMCellModern(output_vocab_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, output_vocab_size) * 0.1
        self.b_out = np.zeros((1, output_vocab_size))

    def encoder_forward(self, X):
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))
        for t in range(X.shape[0]):
            x_t = X[t:t+1, :]
            h, c = self.encoder_cell.forward(x_t, h, c)
        return h, c

    def decoder_forward(self, Y, context):
        h = context
        c = np.zeros((1, self.hidden_size))
        outputs = []
        for t in range(Y.shape[0]):
            y_t = Y[t:t+1, :]
            h, c = self.decoder_cell.forward(y_t, h, c)
            logits = np.dot(h, self.W_out) + self.b_out
            outputs.append(logits)
        return np.array(outputs)

    def forward(self, X, Y):
        context, _ = self.encoder_forward(X)
        logits = self.decoder_forward(Y, context)
        return logits


if __name__ == "__main__":
    print("=== Seq2Seq Modern Demo ===")
    vocab_size = 10
    hidden_size = 16
    seq_len = 5

    model = Seq2SeqModern(vocab_size, vocab_size, hidden_size)
    X = np.random.randn(seq_len, vocab_size)
    Y = np.random.randn(seq_len, vocab_size)

    logits = model.forward(X, Y)
    print(f"Output logits shape: {logits.shape}")