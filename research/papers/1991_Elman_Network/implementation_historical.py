'''
Elman Network (1991) – Historical Implementation (Educational)
Architecture:
- Hidden state feeds back to itself (recurrent connection).
- Uses backpropagation through time (BPTT) for training.
- This version implements a simplified BPTT for demonstration.

Note: The original paper used full BPTT; this educational version
provides a simplified approximation that still captures the core idea.
'''

import numpy as np

class ElmanNetworkHistorical:
    """
    Elman Network (1991) with hidden-state recurrence.
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # Input to hidden
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.1
        # Hidden to hidden (recurrent)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        # Hidden to output
        self.W_ho = np.random.randn(hidden_size, output_size) * 0.1
        self.b_h = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x, h_prev):
        """
        Forward pass for a single time step.
        x: input vector (1 x input_size)
        h_prev: previous hidden state (1 x hidden_size)
        Returns: output, new hidden state
        """
        h = self.sigmoid(np.dot(x, self.W_ih) + np.dot(h_prev, self.W_hh) + self.b_h)
        y = self.sigmoid(np.dot(h, self.W_ho) + self.b_o)
        # Cache for backward pass
        self._cache = (x, h_prev, h, y)
        return y, h

    def backward(self, x, h_prev, target, pred, h, y):
        """
        Simplified BPTT for one time step (educational).
        """
        # Output error
        error = target - pred
        d_y = error * self.sigmoid_derivative(y)

        # Hidden error (backprop from output)
        d_h = np.dot(d_y, self.W_ho.T) * self.sigmoid_derivative(h)

        # Update weights (simplified: uses only current time step)
        self.W_ho += self.lr * np.dot(h.T, d_y)
        self.b_o += self.lr * np.sum(d_y, axis=0, keepdims=True)
        self.W_ih += self.lr * np.dot(x.T, d_h)
        self.W_hh += self.lr * np.dot(h_prev.T, d_h)
        self.b_h += self.lr * np.sum(d_h, axis=0, keepdims=True)

        return np.mean(error ** 2)

    def train_sequence(self, X, y, epochs=100):
        """
        Train on a sequence.
        X: input sequence (timesteps x input_size)
        y: target sequence (timesteps x output_size)
        """
        T = X.shape[0]
        for epoch in range(epochs):
            h = np.zeros((1, self.W_hh.shape[0]))  # initial hidden state = zero
            total_loss = 0.0
            for t in range(T):
                x_t = X[t:t+1, :]
                y_t = y[t:t+1, :]

                # Forward
                h_new = self.sigmoid(np.dot(x_t, self.W_ih) + np.dot(h, self.W_hh) + self.b_h)
                pred = self.sigmoid(np.dot(h_new, self.W_ho) + self.b_o)

                # Backward (simplified BPTT)
                loss = self.backward(x_t, h, y_t, pred, h_new, pred)
                total_loss += loss

                # Update hidden state
                h = h_new

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    def generate(self, X, steps):
        """Generate a sequence of length steps given initial input."""
        h = np.zeros((1, self.W_hh.shape[0]))
        outputs = []
        for t in range(steps):
            x_t = X[t:t+1, :] if t < X.shape[0] else np.zeros((1, X.shape[1]))
            h = self.sigmoid(np.dot(x_t, self.W_ih) + np.dot(h, self.W_hh) + self.b_h)
            y = self.sigmoid(np.dot(h, self.W_ho) + self.b_o)
            outputs.append(y.flatten())
        return np.array(outputs)


if __name__ == "__main__":
    print("=== Elman Network Historical Demo (Educational) ===")
    print("Note: Simplified BPTT implementation for demonstration.")
    timesteps = 50
    X = np.random.randn(timesteps, 1) * 0.1
    y = np.sin(np.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1)

    net = ElmanNetworkHistorical(input_size=1, hidden_size=10, output_size=1, lr=0.01)
    net.train_sequence(X, y, epochs=100)
    print("Training complete.")