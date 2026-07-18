"""
LSTM 1997 – Modern Educational Forward-Pass Implementation

This is a forward-pass-only demonstration of the original 1997 LSTM
(no forget gate). It is designed for clarity and educational purposes.
Full BPTT is not implemented here.
"""

import numpy as np

class LSTMCell_1997_Modern:
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

if __name__ == "__main__":
    print("=== LSTM 1997 Modern Forward-Pass Demo ===")
    cell = LSTMCell_1997_Modern(input_size=2, hidden_size=3)
    x = np.random.randn(1, 2)
    h = np.zeros((1, 3))
    c = np.zeros((1, 3))
    h_new, c_new = cell.forward(x, h, c)
    print(f"Forward pass successful. New hidden shape: {h_new.shape}")