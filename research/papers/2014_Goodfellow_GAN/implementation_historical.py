"""
GAN (2014) – Historical Forward-Pass Demonstration
Architecture:
- Generator: MLP that maps latent noise to generated samples
- Discriminator: MLP that classifies samples as real or fake

Note: This is a forward-pass-only demonstration of the architecture.
Full training with the adversarial minimax objective is not implemented here.
"""

import numpy as np

class Generator:
    """Simple MLP generator."""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, z):
        h = self.relu(np.dot(z, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out  # linear output (real-valued sample)


class Discriminator:
    """Simple MLP discriminator."""
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
        prob = self.sigmoid(logits)
        return prob, logits


class GAN_2014:
    """
    GAN architecture (Goodfellow et al., 2014).
    Generator: learns to generate realistic samples.
    Discriminator: learns to distinguish real from fake.
    """
    def __init__(self, latent_dim=10, data_dim=2, hidden_dim=10):
        self.generator = Generator(latent_dim, hidden_dim, data_dim)
        self.discriminator = Discriminator(data_dim, hidden_dim)
        print("GAN 2014 architecture initialized (forward-pass only).")

    def generate(self, z):
        """Generate a sample from random noise."""
        return self.generator.forward(z)

    def discriminate(self, x):
        """Classify a sample as real or fake."""
        return self.discriminator.forward(x)

    def forward(self, z):
        """Full forward pass: generate and then classify."""
        generated = self.generate(z)
        prob, logits = self.discriminate(generated)
        return generated, prob, logits


if __name__ == "__main__":
    print("=== GAN 2014 Forward-Pass Demo ===")
    print("Note: This is a conceptual demonstration. Full adversarial training")
    print("requires alternating updates to the generator and discriminator.")

    np.random.seed(42)
    gan = GAN_2014(latent_dim=2, data_dim=2, hidden_dim=5)

    # Sample random noise
    z = np.random.randn(1, 2)
    generated, prob, logits = gan.forward(z)

    print(f"Latent vector (z): {z}")
    print(f"Generated sample: {generated}")
    print(f"Discriminator probability (real): {prob[0][0]:.4f}")
    print("Forward pass complete.")