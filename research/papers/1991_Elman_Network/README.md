# 1991_Elman_Network — Finding Structure in Time

**Author:** Jeffrey L. Elman  
**Year:** 1991  
**Venue:** Cognitive Science, Vol. 14, No. 2

---

## Historical Importance

Elman Network is the **standard Simple RNN** that introduced **hidden-state recurrence**, greatly improving upon the Jordan Network. It became the foundation for sequence modelling for over a decade before being superseded by LSTM.

---

## Architecture Diagram

```text
         ┌─────────────────────────────────────────────┐
         │                                             │
         ▼                                             │
  ┌───────────┐    ┌───────────┐    ┌───────────┐     │
  │  Input    │───►│  Hidden   │───►│  Output   │     │
  │  Layer    │    │  Layer    │    │  Layer    │     │
  └───────────┘    └───────────┘    └───────────┘     │
         │               ▲               │             │
         │               │               │             │
         │         ┌─────┴─────┐         │             │
         │         │  Hidden   │─────────┘             │
         │         │  Context  │                       │
         │         │  (state)  │                       │
         │         └───────────┘                       │
         │                                             │
         └─────────────────────────────────────────────┘
```

- **Hidden State:** The network uses the previous hidden state as context.
- **Recurrent Connection:** Hidden → Hidden weight matrix allows temporal information flow.
- **Training:** Backpropagation through time (BPTT) unfolds the network over time.

---

## Key Contributions

- **Hidden‑state recurrence** provides richer context than output feedback.
- **Simplified architecture** compared to Jordan (no separate context units).
- **Demonstrated ability** to learn temporal patterns (e.g., sequence prediction).

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
| `implementation_historical.py` | Historical (educational) implementation |
| `implementation_modern.py` | Modern pedagogical implementation |
| `comparison.md` | Comparison with Jordan Network |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor (Architecture) | 1990_Jordan_Network |
| Predecessor (Learning) | 1986_Rumelhart_Hinton_Williams_Backprop |
| Successor | 1997_LSTM |
| Successor | 2014_Seq2Seq |