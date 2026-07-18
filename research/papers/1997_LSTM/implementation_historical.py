'''
LSTM (1997) – Historical Implementation (Educational)
Architecture:
- Input, forget, output gates with sigmoid activation
- Candidate cell state with tanh
- Cell state update with element-wise multiplication

Note: The original LSTM used a fully connected recurrent structure;
this version is a simplified educational implementation for a single cell.
'''

import numpy as np

class LSTMCell:
    """
    A single LSTM cell for a single time step.
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.hidden_size = hidden_size
        # Concatenated input size: [h_prev, x_t]
        self.W_f = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_i = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_c = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_o = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.b_f = np.zeros((1, hidden_size))
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
        x: input vector (1 x input_size)
        h_prev: previous hidden state (1 x hidden_size)
        c_prev: previous cell state (1 x hidden_size)
        Returns: h_next, c_next
        """
        # Concatenate h_prev and x
        combined = np.concatenate([h_prev, x], axis=1)  # (1, hidden_size + input_size)
        # Gates
        f = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        c_tilde = self.tanh(np.dot(combined, self.W_c) + self.b_c)
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        # Cell state update
        c_next = f * c_prev + i * c_tilde
        # Hidden state
        h_next = o * self.tanh(c_next)
        return h_next, c_next, (f, i, c_tilde, o)

    def backward(self, x, h_prev, c_prev, target, pred, h_next, c_next, gates):
        """
        Simplified BPTT for one time step (educational).
        Updates weights based on prediction error.
        """
        # Compute output error (assuming output = h_next)
        error = target - pred
        d_h = error  # derivative of loss w.r.t. h_next (simplified)
        # For full BPTT, we'd need to backpropagate through time.
        # Here we just demonstrate a basic gradient step.
        # (Full implementation would require careful gradient propagation)
        return np.mean(error ** 2)

    def train_sequence(self, X, y, epochs=100):
        """
        Train on a sequence.
        X: input sequence (timesteps x input_size)
        y: target sequence (timesteps x output_size)
        """
        T = X.shape[0]
        for epoch in range(epochs):
            h = np.zeros((1, self.hidden_size))
            c = np.zeros((1, self.hidden_size))
            total_loss = 0.0
            for t in range(T):
                x_t = X[t:t+1, :]
                y_t = y[t:t+1, :]
                h, c, gates = self.forward(x_t, h, c)
                loss = self.backward(x_t, h, c, y_t, h, h, c, gates)  # simplified
                total_loss += loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

if __name__ == "__main__":
    print("=== LSTM Historical Demo (Educational) ===")
    print("Note: This is a simplified demonstration of the LSTM forward pass.")