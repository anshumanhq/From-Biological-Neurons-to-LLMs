# Long Short-Term Memory

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

## 4. Mathematical Formulation

**Cell State** \(c_t\) is the memory that flows through time with linear update:

```latex
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
```

**Gates** (sigmoid activations):

```latex
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
```
```latex
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
```
```latex
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
```

**Candidate cell state** (tanh):

```latex
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
```

**Hidden state**:

```latex
h_t = o_t \odot \tanh(c_t)
```

---

## 5. Original Paper Analysis

The 1997 paper introduced the **LSTM** architecture with the following innovations:

- **Constant Error Carousel (CEC):** A linear, self-connected unit that allows error to flow unchanged through time (when the forget gate is open).
- **Input and Output Gates:** Control access to the cell state, preventing irrelevant information from being stored or emitted.
- **Forget Gate:** Later added (in the 1999 version, but the 1997 paper already contained the core idea of multiplicative gate units).

The paper demonstrated that LSTM could learn tasks that were impossible for standard RNNs, such as long-range sequence classification and generation.

---

## 6. Algorithm / Method

**Forward Pass (per time step):**

1. Compute concatenated input: `combined = [h_{t-1}, x_t]`
2. Compute gates: `f, i, o` with sigmoid.
3. Compute candidate cell state: `c_tilde` with tanh.
4. Update cell state: `c_t = f * c_{t-1} + i * c_tilde`
5. Update hidden state: `h_t = o * tanh(c_t)`

**Backward Pass (BPTT):**

Unfold the network over time, compute gradients for each gate and the cell state, and propagate errors backward through the linear cell-state recurrence.

---

## 7. NumPy Scratch Implementation

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size, lr=0.01):
        self.lr = lr
        self.hidden_size = hidden_size
        self.W_f = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_i = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_c = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.W_o = np.random.randn(hidden_size + input_size, hidden_size) * 0.1
        self.b_f = np.zeros((1, hidden_size))
        self.b_i = np.zeros((1, hidden_size))
        self.b_c = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        return 1/(1+np.exp(-np.clip(x, -250, 250)))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        combined = np.concatenate([h_prev, x], axis=1)
        f = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        c_tilde = self.tanh(np.dot(combined, self.W_c) + self.b_c)
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        c_next = f * c_prev + i * c_tilde
        h_next = o * self.tanh(c_next)
        return h_next, c_next
```

(For full training, see `implementation_historical.py` and `implementation_modern.py`.)

---

## 8. Limitations (As Acknowledged by Hochreiter & Schmidhuber and Later Research)

- **Complexity:** More parameters and computations than simple RNNs.
- **Overfitting:** LSTM can overfit on small datasets.
- **Hyperparameter Sensitivity:** Learning rate, initialisation, and gate biases require careful tuning.
- **Not a True Solution to All Long-Term Dependencies:** Even LSTM has limits on sequence length.
- **Biological Implausibility:** The gating mechanisms are computationally motivated, not neurobiologically realistic.

---

## 9. Influence on Later Research

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
- Gating controls the flow of information into and out of a memory cell.
- The Constant Error Carousel allows gradients to remain stable over long sequences.

---

## 12. Historical Timeline

- **Before:**
  - 1991: Elman Network
  - 1991: Hochreiter's diploma thesis on vanishing gradients
- **Publication:**
  - 1997: LSTM paper
- **After:**
  - 2014: GRU
  - 2014: Seq2Seq
  - 2017: Transformer

---

## 13. Common Misconceptions

- **Misconception 1:** "LSTM completely solves the vanishing gradient problem."
  - *Fact:* It mitigates it but does not eliminate it for infinite sequences.
- **Misconception 2:** "LSTM was the first gated RNN."
  - *Fact:* The forget gate was introduced later (1999) by Gers et al.
- **Misconception 3:** "LSTM is biologically plausible."
  - *Fact:* It is not.

---

## 14. Implementation Verification

```python
def test_lstm_forward():
    cell = LSTMCell(input_size=2, hidden_size=3)
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
- **Successor:** 2014_Seq2Seq
- **Successor:** 2017_Transformer

---

## 16. Historical Accuracy Check

**Claims in the paper:**
1. LSTM can learn long-range dependencies.
2. The Constant Error Carousel preserves gradient flow.
3. Gating provides control over memory retention.

**Modern interpretation:** Accurate and foundational for sequence modelling.

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

- LSTM was later extended with a **forget gate** by Gers et al. in 1999.
- The paper is one of the most cited in machine learning.
- Hochreiter and Schmidhuber received the 2021 IEEE Neural Networks Pioneer Award.