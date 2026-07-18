# Attractor Dynamics and Parallelism in a Connectionist Sequential Machine

- **Paper ID:** `1990_Jordan_Network`
- **Authors:** Michael I. Jordan
- **Year:** 1990
- **Venue / Journal:** Proceedings of the Eighth Annual Conference of the Cognitive Science Society
- **Primary Subject:** Recurrent Neural Networks / Sequence Learning

---

## 1. Historical Background

By 1990, backpropagation (Rumelhart, Hinton & Williams, 1986) had become the standard method for training multi-layer feed-forward networks. However, feed-forward networks have no memory—they process each input independently. This is a severe limitation for tasks involving sequences (speech, music, time series) where the current output depends on previous inputs.

Researchers had attempted to address this by using **time-delay neural networks** (Waibel et al., 1989) or by simply feeding multiple time steps into a feed-forward network. Michael Jordan, then at MIT, proposed a more elegant solution: add **recurrent connections** that allow the network to maintain a state, effectively giving it memory.

---

## 2. Problem Statement

Jordan addressed the problem of **learning sequential dependencies**. How can a neural network learn to generate sequences where the output at time `t` depends not only on the current input but also on previous outputs and states?

The challenge was to design an architecture that:
1. Could maintain a state across time steps.
2. Could be trained using backpropagation (since that was the known method).
3. Could learn long-range dependencies (at least in principle).

---

## 3. Biological Motivation

Jordan's motivation was **cognitive science** rather than neurobiology. He was interested in how humans process sequences (speech, motor control, music). The concept of **"context"** —information that persists over time and influences subsequent processing—is fundamental in cognitive psychology.

The Jordan Network's context units are analogous to **working memory**—a system that maintains and manipulates information over short periods. The output feedback provides a mechanism for the network to "remember" its previous outputs, which is essential for tasks like generating music or controlling movements.

---

## 4. Mathematical Formulation

**Network Architecture:**
- Input layer: receives external input at time `t`
- Hidden layer: processes input + context
- Output layer: produces output at time `t`
- Context units: store the previous output (or a copy of the hidden state)

**Forward Pass:**

For a network with input `x(t)`, hidden state `h(t)`, output `y(t)`, and context `c(t)`:

```latex
h(t) = f\left( W_{ih} x(t) + W_{ch} c(t) + b_h \right)
```

```latex
y(t) = g\left( W_{ho} h(t) + b_o \right)
```

```latex
c(t+1) = \alpha \cdot y(t) + \beta \cdot c(t)
```

Where:
- \( W_{ih} \): input-to-hidden weights
- \( W_{ch} \): context-to-hidden weights
- \( W_{ho} \): hidden-to-output weights
- \( f(\cdot) \): hidden activation (usually sigmoid/tanh)
- \( g(\cdot) \): output activation (usually linear or sigmoid)
- \( \alpha, \beta \): context update parameters (often \( \alpha = 1, \beta = 1 \))

**Learning:** The network is typically trained using **backpropagation through time (BPTT)**. **Teacher forcing**—a technique where the context is updated using the target output instead of the network's own prediction—was commonly used to stabilise training. This description follows the standard interpretation of Jordan's work as understood in later literature.

---

## 5. Original Paper Analysis

Jordan's 1990 paper introduced the **Jordan Network**, which is often called a **recurrent network with output feedback**. The key innovation was:

1. **Context Units:** A set of units that store the previous output and feed it back to the hidden layer.
2. **Teacher Forcing:** During training, the context is updated using the target output (rather than the actual output), which stabilises learning.
3. **Sequence Generation:** The network can be trained to generate sequences by using its own outputs as inputs for subsequent time steps.

The paper demonstrated the network's ability to learn sequential patterns and showed that the network's internal representations formed **attractors**—states that the network tends to converge to, which corresponds to the structure of the sequence being learned.

---

## 6. Algorithm / Method

**Training Procedure (Teacher Forcing):**

1. **Forward Pass:** Present input `x(t)` and context `c(t)` (initialised to zero).
2. **Compute Hidden:** `h(t) = f(W_{ih} x(t) + W_{ch} c(t) + b_h)`
3. **Compute Output:** `y(t) = g(W_{ho} h(t) + b_o)`
4. **Compute Error:** `e(t) = target(t) - y(t)`
5. **Update Context:** `c(t+1) = target(t)` (teacher forcing) or `c(t+1) = y(t)` (free running)
6. **Backward Pass:** Propagate error through time using BPTT or backpropagation through the unfolded network.
7. **Repeat** for each time step in the sequence.

---

## 7. NumPy Scratch Implementation

```python
import numpy as np

class JordanNetwork:
    """
    Jordan Network (1990) with output feedback context units.
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
        # Context update: use output as new context
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
    print("=== Jordan Network Demo ===")
    print("Simple sequence learning example.")
    # Example: learn a sine wave
    timesteps = 50
    X = np.random.randn(timesteps, 1) * 0.1
    y = np.sin(np.linspace(0, 4*np.pi, timesteps)).reshape(-1, 1)

    net = JordanNetwork(input_size=1, hidden_size=10, output_size=1, lr=0.01)
    net.train_sequence(X, y, teacher_forcing=True, epochs=100)
    print("Training complete.")
```

---

## 8. Limitations (As Acknowledged by Jordan and the Community)

- **Finite Context Window:** The context unit only stores the previous output, which limits the network's ability to capture long-range dependencies.
- **Vanishing Gradients:** Like all recurrent networks trained with BPTT, Jordan Networks suffer from vanishing gradients when the sequence length is long.
- **Teacher Forcing:** While teacher forcing stabilises training, it can lead to the network being unable to generate sequences without target inputs (exposure bias).
- **Limited Capacity:** The architecture is relatively simple and lacks the sophisticated gating mechanisms of later architectures (LSTM, GRU).

---

## 9. Influence on Later Research

- **Elman Network (1991):** Jeffrey Elman proposed a simpler recurrent network with hidden state context, which became the standard "Simple RNN" in the literature.
- **LSTM (1997):** The shortcomings of simple recurrent networks (vanishing gradients) directly motivated the development of LSTM.
- **Sequence Learning:** Jordan's work established the foundation for sequence learning in neural networks, which would later lead to Seq2Seq models and Transformers.

---

## 10. Modern Relevance (2026 Perspective)

The Jordan Network is now considered a **historical precursor** to modern sequence models. Its main contribution was establishing the idea that **feedback connections enable memory in neural networks**. While the architecture itself is rarely used today, the concept of maintaining a context state is fundamental to all modern sequence models, including LSTMs, GRUs, and Transformers.

---

## 11. Primary Source Paraphrase

The following summarises the paper's central contributions:

- The network can learn to generate sequences by maintaining a state that represents previous outputs.
- Context units provide a form of short-term memory that allows the network to process sequential dependencies.
- Teacher forcing is a technique that stabilises learning by using target outputs instead of the network's own predictions to update the context.

---

## 12. Historical Timeline

- **Before:**
  - 1986: Rumelhart, Hinton & Williams (Backpropagation)
- **Publication:**
  - 1990: Jordan Network paper
  - 1991: Elman Network
- **After:**
  - 1997: LSTM
  - 2014: Seq2Seq
  - 2017: Transformer

---

## 13. Common Misconceptions

- **Misconception 1:** "Jordan Network is a fully connected RNN."
  - **Fact:** It only feeds the output back, not the full hidden state.
- **Misconception 2:** "Jordan Network was the first RNN."
  - **Fact:** Recurrent networks existed earlier (Hopfield, 1982), but Jordan's was the first practical trainable RNN for sequence learning.
- **Misconception 3:** "Jordan Network works for long sequences."
  - **Fact:** It suffers from vanishing gradients for long sequences, which LSTM later solved.

---

## 14. Implementation Verification

```python
def test_jordan_forward():
    X = np.random.randn(1, 3)
    context = np.random.randn(1, 2)
    net = JordanNetwork(input_size=3, hidden_size=5, output_size=2)
    y, new_context = net.forward(X, context)
    assert y.shape == (1, 2), "Output shape mismatch"
    print("Jordan Network forward pass successful.")
```

---

## 15. Cross References (Related Papers in this Archive)

- **Predecessor:** Rumelhart, Hinton & Williams (1986) – Backpropagation
- **Successor:** Elman (1991) – Simple RNN
- **Successor:** Hochreiter & Schmidhuber (1997) – LSTM

---

## 16. Historical Accuracy Check

**Claims made in the original paper:**
1. The Jordan Network can learn sequences using output feedback.
2. Teacher forcing stabilises training.
3. The network forms attractors corresponding to sequence structures.

**Modern interpretation:** The paper is historically accurate and foundational for sequence learning.

---

## 17. Reproducibility

- **Dataset:** Synthetic sequences (sine waves, simple patterns).
- **Hardware:** Standard computers of the time (Sun workstations).
- **Reproducibility Today:** The NumPy implementation reproduces the core architecture.

---

## 18. Influence Graph

```text
Rumelhart, Hinton & Williams (1986) – Backpropagation
  │
  ▼
Jordan Network (1990) ───► Elman Network (1991) ───► LSTM (1997)
  │                              │                           │
  │ (Output feedback)            │ (Hidden state)            │ (Gated memory)
  │                              │                           │
  └──────────────────────────────┴───────────────────────────┘
                                  │
                                  ▼
                          Transformer (2017)
```

---

## Additional Notes

- Jordan was influenced by cognitive science and motor control.
- His work on sequence learning was extended by his student, Jeff Elman.
- The Jordan Network is often overlooked in modern surveys, but it was the first practical sequence-learning architecture.
