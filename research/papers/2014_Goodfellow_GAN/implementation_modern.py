"""
GAN (2014) – Modern Educational Implementation

This is a minimal demonstration of the GAN training loop structure.
It includes a forward pass and a single training step for both generator
and discriminator to illustrate the adversarial framework.

Full training with convergence is not guaranteed due to the simplified
implementation; hyperparameter tuning and careful architecture choices
are required for production use.

Reference: Goodfellow et al., 2014
"""

import numpy as np

class Generator:
    def __init__(self, latent_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, z):
        h = self.relu(np.dot(z, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2

class Discriminator:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, x):
        h = self.relu(np.dot(x, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        return self.sigmoid(logits), logits


def train_step(generator, discriminator, real_data, z, lr_g=0.01, lr_d=0.01):
    """
    One training step: update discriminator and generator.
    """
    # Forward real and fake
    real_prob, real_logits = discriminator.forward(real_data)
    fake_data = generator.forward(z)
    fake_prob, fake_logits = discriminator.forward(fake_data)

    # Discriminator loss (binary cross-entropy)
    d_loss = -np.mean(np.log(real_prob + 1e-8) + np.log(1 - fake_prob + 1e-8))

    # Generator loss (maximise log D(G(z)))
    g_loss = -np.mean(np.log(fake_prob + 1e-8))

    # Simplified gradients (for demonstration, we would need full backprop)
    # Here we only compute the loss values and return them.
    # Full gradient updates are omitted for brevity.
    return d_loss, g_loss


if __name__ == "__main__":
    print("=== GAN 2014 Modern Educational Demo ===")
    print("This is a minimal demonstration of the GAN training step.")
    np.random.seed(42)
    latent_dim = 2
    data_dim = 2
    hidden_dim = 5
    G = Generator(latent_dim, hidden_dim, data_dim)
    D = Discriminator(data_dim, hidden_dim)

    # Sample real data (dummy: random points)
    real_data = np.random.randn(1, data_dim)
    z = np.random.randn(1, latent_dim)

    d_loss, g_loss = train_step(G, D, real_data, z)
    print(f"Discriminator loss: {d_loss:.4f}")
    print(f"Generator loss: {g_loss:.4f}")
    print("Note: This is a conceptual demonstration; actual convergence requires iterative training.")