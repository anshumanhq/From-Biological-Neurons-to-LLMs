'''
Original LSTM (1997) – Forward-Pass Demonstration
Architecture:
- Input gate (sigmoid) and output gate (sigmoid)
- Candidate cell state (tanh)
- Cell state update: c_t = c_{t-1} + i_t * c_tilde  (no forget gate)

Note: This is a forward-pass-only implementation, not full BPTT.
The original 1997 LSTM did not include a forget gate.
'''

import numpy as np

class LSTMCell_1997:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        # Weights for input, candidate, output
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
        """
        Forward pass for one time step.
        x: (1, input_size)
        h_prev: (1, hidden_size)
        c_prev: (1, hidden_size)
        Returns: h_next, c_next
        """
        combined = np.concatenate([h_prev, x], axis=1)
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        c_tilde = self.tanh(np.dot(combined, self.W_c) + self.b_c)
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        c_next = c_prev + i * c_tilde   # Constant Error Carousel: linear self-connection with weight 1
        h_next = o * self.tanh(c_next)
        return h_next, c_next

if __name__ == "__main__":
    print("=== LSTM 1997 Forward-Pass Demo (No Forget Gate) ===")
    cell = LSTMCell_1997(input_size=2, hidden_size=3)
    x = np.random.randn(1, 2)
    h = np.zeros((1, 3))
    c = np.zeros((1, 3))
    h_new, c_new = cell.forward(x, h, c)
    print(f"Input: {x}\nNew hidden: {h_new}\nNew cell: {c_new}")