# 1990_Jordan_Network — Attractor Dynamics and Parallelism in a Connectionist Sequential Machine

**Authors:** Michael I. Jordan  
**Year:** 1990  
**Venue:** Proceedings of the Eighth Annual Conference of the Cognitive Science Society

---

## Historical Importance

This paper introduced the **Jordan Network**, the first practical recurrent neural network architecture for sequence learning. It established the concept of **context units**—feedback connections from the output layer to the hidden layer—that give the network a form of short-term memory.

---

## Architecture Diagram

'''text
         ┌─────────────────────────────────────────────┐
         │                                             │
         ▼                                             │
  ┌───────────┐    ┌───────────┐    ┌───────────┐     │
  │  Input    │───►│  Hidden   │───►│  Output   │─────┘
  │  Layer    │    │  Layer    │    │  Layer    │
  └───────────┘    └───────────┘    └───────────┘
         │               ▲                │
         │               │                │
         │         ┌─────┴─────┐          │
         │         │  Context  │◄─────────┘
         │         │  Units    │
         │         └───────────┘
         │               │
         └───────────────┘
'''

- **Context Units:** Store the previous output and feed it back to the hidden layer.
- **Teacher Forcing:** During training, the context is updated with the target output (stabilises learning).
- **Free Running:** During generation, the context is updated with the network's own output.

---

## Key Contributions

- **Context Units:** Store previous output and feed it back to the hidden layer.
- **Teacher Forcing:** A training technique that stabilises learning by using target outputs as context.
- **Sequence Generation:** Demonstrated the network's ability to generate sequences using its own outputs.

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
| `metadata.yaml` | Structured metadata for automation |
| `paper_source.md` | DOI, publisher, access notes |
| `implementation_historical.py` | Historical implementation (1990 style) |
| `implementation_modern.py` | Modern pedagogical implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor (Learning) | 1986_Rumelhart_Hinton_Williams_Backprop |
| Successor | 1991_Elman_Network |
| Successor | 1997_LSTM |