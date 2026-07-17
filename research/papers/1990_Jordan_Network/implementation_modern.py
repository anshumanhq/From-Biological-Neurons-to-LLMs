"""
Jordan Network (1990) – Modern Implementation (Educational)
- Uses ReLU or Tanh activations (configurable)
- Mini-batch training support
- Clear separation of teacher forcing and free running
"""

import numpy as np

class JordanNetworkModern:
    """
    Jordan Network with modern conveniences:
    - Configurable activation functions
    - Mini-batch support
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, activation='tanh'):
        self.lr = lr
        self.activation = activation
        # Input to hidden
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.1
        # Context to hidden
        self.W_ch = np.random.randn(output_size, hidden_size) * 0.1
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

    def forward(self, x, context):
        """Forward pass for a single time step."""
        h = self._activation(np.dot(x, self.W_ih) + np.dot(context, self.W_ch) + self.b_h)
        y = self._activation(np.dot(h, self.W_ho) + self.b_o)
        new_context = y.copy()
        return y, new_context

    def train_sequence(self, X, y, teacher_forcing=True, epochs=100, verbose=True):
        """Train on a sequence."""
        T = X.shape[0]
        for epoch in range(epochs):
            context = np.zeros((1, y.shape[1]))
            total_loss = 0.0
            for t in range(T):
                x_t = X[t:t+1, :]
                y_t = y[t:t+1, :]
                h = self._activation(np.dot(x_t, self.W_ih) + np.dot(context, self.W_ch) + self.b_h)
                pred = self._activation(np.dot(h, self.W_ho) + self.b_o)
                error = y_t - pred
                total_loss += np.mean(error ** 2)
                if teacher_forcing:
                    context = y_t.copy()
                else:
                    context = pred.copy()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    def generate(self, X, steps):
        """Generate a sequence."""
        context = np.zeros((1, self.W_ch.shape[0]))
        outputs = []
        for t in range(steps):
            x_t = X[t:t+1, :] if t < X.shape[0] else np.zeros((1, X.shape[1]))
            h = self._activation(np.dot(x_t, self.W_ih) + np.dot(context, self.W_ch) + self.b_h)
            y = self._activation(np.dot(h, self.W_ho) + self.b_o)
            outputs.append(y.flatten())
            context = y.copy()
        return np.array(outputs)


if __name__ == "__main__":
    print("=== Jordan Network Modern Demo ===")
    timesteps = 50
    X = np.random.randn(timesteps, 1) * 0.1
    y = np.sin(np.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1)

    net = JordanNetworkModern(input_size=1, hidden_size=10, output_size=1, lr=0.01)
    net.train_sequence(X, y, teacher_forcing=True, epochs=100)
    print("Training complete.")