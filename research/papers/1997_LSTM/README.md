# 1997_LSTM — Long Short-Term Memory (Original)

**Authors:** Sepp Hochreiter, Jürgen Schmidhuber  
**Year:** 1997  
**Venue:** Neural Computation, Vol. 9, No. 8

---

## Historical Importance

The original 1997 LSTM introduced the **Constant Error Carousel (CEC)** and **gated input/output** mechanisms, enabling reliable learning of long-term dependencies. **This version did not include a forget gate** (that was added in 1999/2000).

---

## Architecture Diagram (1997)

```text
         Input     Output
          Gate      Gate
            |         |
    x_t ----+---------+----> h_t
            |         |
            v         v
          ┌─────────────────────┐
          │   LSTM Cell         │
          │                     │
          │   c_t = c_{t-1}     │
          │        + i_t*c_tilde│
          │                     │
          │   h_t = o_t*tanh(c_t)│
          └─────────────────────┘
                 ▲      ▲
                 |      |
               c_{t-1}  h_{t-1}
```

- **Cell State** `c_t` updates additively: `c_t = c_{t-1} + i_t * c_tilde`.
- **No forget gate** – the memory never resets (adaptive resetting came later).

---

## Key Contributions

- **Constant Error Carousel** – gradient flow preserved over long time intervals.
- **Input/Output gating** – controls what is stored and emitted.
- **Long-range dependency learning** – demonstrated on tasks >100 time steps.

---

## Available Files

| File | Description |
| :--- | :--- |
| `notes.md` | Full 18‑section historical analysis |
| `summary.md` | One‑page abstract |
| `equations.tex` | Core LaTeX equations (no forget gate) |
| `bibliography.bib` | BibTeX entry |
| `timeline.md` | Historical timeline context |
| `questions.md` | Open questions and debates |
| `metadata.yaml` | Structured metadata |
| `paper_source.md` | DOI, publisher, access notes |
| `implementation_historical.py` | Forward-pass demo (no training) |
| `implementation_modern.py` | Modern educational forward stub |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor (RNN) | 1991_Elman_Network |
| Successor (Forget Gate) | 1999_LSTM_ForgetGate |
| Successor | 2014_Seq2Seq |
| Successor | 2017_Transformer |