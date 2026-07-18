# 1997_LSTM — Long Short-Term Memory

**Authors:** Sepp Hochreiter, Jürgen Schmidhuber  
**Year:** 1997  
**Venue:** Neural Computation, Vol. 9, No. 8

---

## Historical Importance

LSTM is the most influential recurrent architecture before Transformers. It introduced the **gated cell state** and **Constant Error Carousel**, which allow the network to preserve gradients over hundreds of time steps. This made it possible to train deep RNNs for tasks like speech recognition, machine translation, and handwriting recognition.

---

## Architecture Diagram

```text
         Input     Forget    Output
          Gate      Gate      Gate
            |         |         |
    x_t ----+---------+---------+----> h_t
            |         |         |
            v         v         v
          ┌─────────────────────┐
          │   LSTM Cell         │
          │                     │
          │   c_t = f_t*c_{t-1} │
          │        + i_t*c_tilde│
          │                     │
          │   h_t = o_t*tanh(c_t)│
          └─────────────────────┘
                 ▲      ▲
                 |      |
               c_{t-1}  h_{t-1}
```

- **Cell State** `c_t` is the memory that flows through time with minimal transformation.
- **Gates** control what to forget, what to store, and what to output.

---

## Key Contributions

- **Constant Error Carousel** – allows error to flow unchanged through the cell state.
- **Gated Architecture** – learn when to remember and when to forget.
- **Long-range Dependency Learning** – enables modelling of sequences with >100 steps.

---

## Available Files

| File | Description |
| :--- | :--- |
| `notes.md` | Full 18‑section historical analysis |
| `summary.md` | One‑page abstract |
| `equations.tex` | Core LaTeX equations |
| `bibliography.bib` | BibTeX entry |
| `timeline.md` | Historical timeline context |
| `questions.md` | Open questions and debates |
| `metadata.yaml` | Structured metadata |
| `paper_source.md` | DOI, publisher, access notes |
| `implementation_historical.py` | Simplified educational forward pass |
| `implementation_modern.py` | Modern pedagogical stub (with NotImplementedError) |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor (RNN) | 1991_Elman_Network |
| Predecessor (RNN) | 1990_Jordan_Network |
| Successor | 2014_Seq2Seq |
| Successor | 2017_Transformer |