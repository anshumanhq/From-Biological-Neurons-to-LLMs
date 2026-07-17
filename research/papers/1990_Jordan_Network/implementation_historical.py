'''
Jordan Network (1990) – Historical Implementation
Faithful to the original architecture:
- Output feedback context units
- Teacher forcing for training
- Sigmoid activation
'''

import numpy as np

class JordanNetworkHistorical:
    """
    Jordan Network (1990) with output feedback context units.
    Uses teacher forcing during training.
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # Input to hidden
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.1
        # Context to hidden
        self.W_ch = np.random.randn(output_size, hidden_size) * 0.1
        # Hidden to output
        self.W_ho = np.random.randn(hidden_size, output_size) * 0.1
        self.b_h = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, x, context):
        """
        Forward pass for a single time step.
        x: input vector (1 x input_size)
        context: context vector (1 x output_size)
        Returns: output, new_context
        """
        h = self.sigmoid(np.dot(x, self.W_ih) + np.dot(context, self.W_ch) + self.b_h)
        y = self.sigmoid(np.dot(h, self.W_ho) + self.b_o)
        new_context = y.copy()
        return y, new_context

    def train_sequence(self, X, y, teacher_forcing=True, epochs=100):
        """
        Train on a sequence.
        X: input sequence (timesteps x input_size)
        y: target sequence (timesteps x output_size)
        """
        T = X.shape[0]
        for epoch in range(epochs):
            context = np.zeros((1, y.shape[1]))
            total_loss = 0.0
            for t in range(T):
                x_t = X[t:t+1, :]
                y_t = y[t:t+1, :]
                # Forward
                h = self.sigmoid(np.dot(x_t, self.W_ih) + np.dot(context, self.W_ch) + self.b_h)
                pred = self.sigmoid(np.dot(h, self.W_ho) + self.b_o)
                # Loss
                error = y_t - pred
                total_loss += np.mean(error ** 2)
                # Update context (teacher forcing or free running)
                if teacher_forcing:
                    context = y_t.copy()
                else:
                    context = pred.copy()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    def generate(self, X, steps):
        """Generate a sequence of length steps given initial input."""
        context = np.zeros((1, self.W_ch.shape[0]))
        outputs = []
        for t in range(steps):
            x_t = X[t:t+1, :] if t < X.shape[0] else np.zeros((1, X.shape[1]))
            h = self.sigmoid(np.dot(x_t, self.W_ih) + np.dot(context, self.W_ch) + self.b_h)
            y = self.sigmoid(np.dot(h, self.W_ho) + self.b_o)
            outputs.append(y.flatten())
            context = y.copy()
        return np.array(outputs)


if __name__ == "__main__":
    print("=== Jordan Network Historical Demo ===")
    # Simple sequence: learn a sine wave
    timesteps = 50
    X = np.random.randn(timesteps, 1) * 0.1
    y = np.sin(np.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1)

    net = JordanNetworkHistorical(input_size=1, hidden_size=10, output_size=1, lr=0.01)
    net.train_sequence(X, y, teacher_forcing=True, epochs=100)
    print("Training complete.")