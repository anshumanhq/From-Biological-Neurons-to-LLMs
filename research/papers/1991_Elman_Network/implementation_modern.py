"""
Elman Network (1991) – Modern Implementation (Educational)
- Uses tanh or ReLU activation (configurable)
- This version intentionally omits the training loop; see implementation_historical.py.
- Provided for architecture clarity and forward pass demonstration.
"""

import numpy as np

class ElmanNetworkModern:
    """
    Elman Network with configurable activation.
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, activation='tanh'):
        self.lr = lr
        self.activation = activation
        # Input to hidden
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.1
        # Hidden to hidden (recurrent)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        # Hidden to output
        self.W_ho = np.random.randn(hidden_size, output_size) * 0.1
        self.b_h = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, output_size))

    def _activation(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        else:  # relu
            return np.maximum(0, x)

    def forward(self, x, h_prev):
        """
        Forward pass for a single time step.
        """
        h = self._activation(np.dot(x, self.W_ih) + np.dot(h_prev, self.W_hh) + self.b_h)
        y = self._activation(np.dot(h, self.W_ho) + self.b_o)
        return y, h

    def train_sequence(self, X, y, epochs=100, verbose=True):
        """
        Training loop intentionally omitted for clarity.
        Please use implementation_historical.py for an educational BPTT loop.
        """
        raise NotImplementedError(
            "Training loop not implemented in modern version. "
            "See implementation_historical.py for a simplified educational BPTT."
        )

    def generate(self, X, steps):
        """Generate a sequence."""
        h = np.zeros((1, self.W_hh.shape[0]))
        outputs = []
        for t in range(steps):
            x_t = X[t:t+1, :] if t < X.shape[0] else np.zeros((1, X.shape[1]))
            h = self._activation(np.dot(x_t, self.W_ih) + np.dot(h, self.W_hh) + self.b_h)
            y = self._activation(np.dot(h, self.W_ho) + self.b_o)
            outputs.append(y.flatten())
        return np.array(outputs)


if __name__ == "__main__":
    print("=== Elman Network Modern Demo (Forward Pass Only) ===")
    net = ElmanNetworkModern(input_size=2, hidden_size=4, output_size=1)
    x = np.random.randn(1, 2)
    h = np.zeros((1, 4))
    y, h_new = net.forward(x, h)
    print(f"Input: {x}, Output: {y}, New hidden: {h_new}")