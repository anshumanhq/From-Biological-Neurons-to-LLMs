# Comparison: Elman Network (1991) vs. Jordan Network (1990)

## Overview

Both networks are early recurrent architectures that enabled sequence learning. The key difference lies in **what they feed back as context**.

| Aspect | Jordan (1990) | Elman (1991) |
| :--- | :--- | :--- |
| **Context source** | Previous **output** `y(t-1)` | Previous **hidden state** `h(t-1)` |
| **Recurrent connection** | Output → Context → Hidden | Hidden → Hidden |
| **Information richness** | Limited (only output) | Richer (full hidden representation) |
| **Training** | BPTT + teacher forcing | BPTT (teacher forcing optional) |
| **Biological inspiration** | Cognitive science (working memory) | Cognitive science (language) |
| **Main application** | Motor control, simple sequences | Language, complex temporal patterns |
| **Historical role** | Proof-of-concept for recurrence | **Standard Simple RNN** |

---

## Why Hidden-State Context is Better

1. **Richer Representation:** The hidden state captures all prior information, not just the output. It can maintain a more detailed summary of the past.
2. **Greater Expressivity:** The hidden state can represent complex internal dynamics, while output feedback is limited by the output dimension.
3. **Unified Architecture:** Elman’s network is simply a feed-forward network with a recurrence from hidden to hidden; no separate context units are needed.

---

## Performance Comparison (Illustrative)

- On simple tasks (e.g., sine wave prediction), both perform similarly.
- On language tasks (e.g., predicting the next word in a sentence), Elman outperforms Jordan due to richer contextual representation.

---

## Limitations of Both

- **Vanishing Gradients:** Both suffer from the vanishing gradient problem when sequences are long.
- **Capacity:** Single-hidden-layer RNNs have limited representational power.
- **Exposure Bias:** Teacher forcing can cause issues during generation.

---

## Legacy

Elman became the **de facto Simple RNN** taught in textbooks. Jordan is often referenced as a predecessor. The step from Jordan to Elman is a classic example of architectural refinement that motivated further research, eventually leading to LSTMs and Transformers.