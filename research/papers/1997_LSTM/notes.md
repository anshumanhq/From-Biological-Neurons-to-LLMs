# Long Short-Term Memory (1997)

- **Paper ID:** `1997_LSTM`
- **Authors:** Sepp Hochreiter, Jürgen Schmidhuber
- **Year:** 1997
- **Venue / Journal:** *Neural Computation*, Vol. 9, No. 8, pp. 1735–1780
- **DOI:** 10.1162/neco.1997.9.8.1735
- **Primary Subject:** Recurrent Neural Networks / Sequence Learning / Gradient-based Optimization

---

## 1. Historical Background

By 1997, researchers had established that simple recurrent networks (Elman, Jordan) could learn short-term dependencies, but they struggled with long sequences. The **vanishing gradient problem** (Hochreiter, 1991) made it impossible for standard RNNs to capture dependencies that span more than a few time steps. Hochreiter and Schmidhuber proposed a radical solution: design a **linear recurrence** that allows error to flow unchanged through time, supplemented by **gated control** over memory access. This became the Long Short-Term Memory (LSTM) architecture.

---

## 2. Problem Statement

The fundamental problem was the **vanishing gradient** during backpropagation through time (BPTT). Gradients either shrink exponentially (vanishing) or grow exponentially (exploding). The former prevents learning long-range dependencies; the latter makes training unstable. LSTM was designed to **preserve gradient flow** over long intervals by using a **Constant Error Carousel (CEC)**.

---

## 3. Biological Motivation

LSTM was not directly inspired by biology. However, the idea of a memory cell that can selectively retain or forget information bears a loose resemblance to synaptic plasticity and neuromodulatory gating in the brain. The primary motivation was computational: to overcome the mathematical limitations of gradient-based learning in recurrent networks.

---

## 4. Mathematical Formulation (Original 1997)

**Cell State** \(c_t\) updates additively, with **no forget gate**:

```latex
c_t = c_{t-1} + i_t \odot \tilde{c}_t
```

**Input gate** (sigmoid):

```latex
i_t = \sigma\left( W_i \cdot [h_{t-1}, x_t] + b_i \right)
```

**Candidate cell state** (tanh):

```latex
\tilde{c}_t = \tanh\left( W_c \cdot [h_{t-1}, x_t] + b_c \right)
```

**Output gate** (sigmoid):

```latex
o_t = \sigma\left( W_o \cdot [h_{t-1}, x_t] + b_o \right)
```

**Hidden state**:

```latex
h_t = o_t \odot \tanh(c_t)
```

**Constant Error Carousel**: because the cell state update is linear with coefficient 1, the gradient flows back as:

```latex
\frac{\partial c_t}{\partial c_{t-1}} = 1 \quad \text{(idealised, when no external interference)}
```

This is the key mechanism that prevents vanishing gradients.

---

## 5. Original Paper Analysis

The 1997 paper introduced the **LSTM** architecture with the following innovations:

- **Constant Error Carousel (CEC):** A linear, self-connected unit that allows error to flow unchanged through time.
- **Input and Output Gates:** Control access to the cell state, preventing irrelevant information from being stored or emitted.
- **No Forget Gate:** The original memory cell does not have a mechanism to reset its internal state; the CEC always retains information.

The paper demonstrated that LSTM could learn tasks that were impossible for standard RNNs, such as long-range sequence classification and generation.

---

## 6. Algorithm / Method

**Forward Pass (per time step):**

1. Compute concatenated input: `combined = [h_{t-1}, x_t]`
2. Compute input gate: `i = sigmoid(W_i * combined + b_i)`
3. Compute candidate cell state: `c_tilde = tanh(W_c * combined + b_c)`
4. Update cell state: `c_t = c_{t-1} + i * c_tilde`
5. Compute output gate: `o = sigmoid(W_o * combined + b_o)`
6. Compute hidden state: `h_t = o * tanh(c_t)`

**Backward Pass (BPTT):**

Unfold the network over time, compute gradients for each gate and the cell state, and propagate errors backward through the linear cell-state recurrence.

---

## 7. NumPy Scratch Implementation

```python
import numpy as np

class LSTMCell_1997:
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
        c_next = c_prev + i * c_tilde   # no forget gate
        h_next = o * self.tanh(c_next)
        return h_next, c_next
```

The full training loop (BPTT) is not included in this archive; the implementation above is a forward-pass demonstration.

---

## 8. Limitations (As Acknowledged by Hochreiter & Schmidhuber and Later Research)

- **Complexity:** More parameters and computations than simple RNNs.
- **Overfitting:** LSTM can overfit on small datasets.
- **Hyperparameter Sensitivity:** Learning rate, initialisation, and gate biases require careful tuning.
- **No Built-in Reset:** The original LSTM cannot forget; memory accumulates indefinitely unless manually managed.
- **Not a True Solution to All Long-Term Dependencies:** Even LSTM has limits on sequence length.

---

## 9. Influence on Later Research

- **Forget-gate LSTM (1999/2000):** Gers, Schmidhuber & Cummins added an adaptive forget gate to reset the cell state.
- **GRU (2014):** Simplified gated recurrent unit (Cho et al.)
- **Seq2Seq (2014):** LSTM-based encoder-decoder for translation.
- **Attention (2014):** Bahdanau attention used LSTM in the encoder.
- **Transformer (2017):** Replaced LSTM entirely with self-attention, but LSTM influenced the concept of state memory.

---

## 10. Modern Relevance (2026 Perspective)

LSTM is still used in many applications where sequence length is moderate and memory efficiency matters. However, Transformers have largely replaced LSTM in NLP. The core ideas of gating and linear recurrence have influenced modern architectures, including gated linear units and state-space models.

---

## 11. Primary Source Paraphrase

- The paper introduces an architecture that can learn long-range dependencies.
- The Constant Error Carousel allows gradients to remain stable over long sequences.
- Input and output gating controls access to the memory cell.
- The original 1997 LSTM does **not** include a forget gate; adaptive resetting was added later.

---

## 12. Historical Timeline

- **Before:**
  - 1991: Hochreiter's thesis identifies vanishing gradients
  - 1991: Elman Network
- **Publication:**
  - 1997: Original LSTM (CEC + input/output gates)
- **After:**
  - 1999: Gers, Schmidhuber & Cummins introduce the forget gate (local)
  - 2000: Gers et al. publish "Learning to Forget" in Neural Computation
  - 2014: GRU
  - 2014: Seq2Seq
  - 2017: Transformer

---

## 13. Common Misconceptions

- **Misconception 1:** "LSTM had a forget gate from the beginning."
  - **Fact:** The original 1997 LSTM had no forget gate; it was added later (1999/2000).
- **Misconception 2:** "LSTM completely solves the vanishing gradient problem."
  - **Fact:** It mitigates it but does not eliminate it for infinite sequences.
- **Misconception 3:** "LSTM is biologically plausible."
  - **Fact:** It is not.

---

## 14. Implementation Verification

```python
def test_lstm_forward():
    cell = LSTMCell_1997(input_size=2, hidden_size=3)
    x = np.random.randn(1, 2)
    h = np.zeros((1, 3))
    c = np.zeros((1, 3))
    h_new, c_new = cell.forward(x, h, c)
    assert h_new.shape == (1, 3)
    assert c_new.shape == (1, 3)
    print("LSTM forward pass successful.")
```

---

## 15. Cross References

- **Predecessor:** 1991_Elman_Network
- **Predecessor:** 1990_Jordan_Network
- **Successor (Forget Gate):** 1999_LSTM_ForgetGate (added later)
- **Successor:** 2014_Seq2Seq
- **Successor:** 2017_Transformer

---

## 16. Historical Accuracy Check

**Claims in the paper:**
1. LSTM can learn long-range dependencies.
2. The Constant Error Carousel preserves gradient flow.
3. Gating provides control over memory retention.
4. The original 1997 LSTM does **not** have a forget gate.

**Modern interpretation:** Accurate and foundational for sequence modelling. The later addition of the forget gate (1999/2000) became the standard LSTM known today.

---

## 17. Reproducibility

- **Dataset:** Synthetic and real tasks (e.g., speech, handwriting).
- **Hardware:** Standard workstations of the era.
- **Reproducibility Today:** Easily reproducible with NumPy/PyTorch.

---

## 18. Influence Graph

```text
Elman (1991) ──────────────────► LSTM (1997) ──────────────────► Seq2Seq (2014)
   (vanishing gradient)              (gated memory)               (encoder-decoder)
          │                                    │
          │                                    │
          └───────────────┬────────────────────┘
                          │
                          ▼
                   Transformer (2017)
                          │
                          ▼
                      LLMs (2020+)
```

---

## Additional Notes

- LSTM was later extended with a **forget gate** by Gers et al. in 1999 and formally published in 2000.
- The paper is one of the most cited in machine learning.
- Hochreiter and Schmidhuber received the 2021 IEEE Neural Networks Pioneer Award.