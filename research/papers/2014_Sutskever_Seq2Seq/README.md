# 2014_Sutskever_Seq2Seq — Sequence to Sequence Learning with Neural Networks

**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le  
**Year:** 2014  
**Venue:** NeurIPS (Advances in Neural Information Processing Systems)

---

## Historical Importance

Seq2Seq introduced the **encoder-decoder architecture** for sequence transduction tasks like neural machine translation. It used two LSTMs: one to encode the source sequence into a fixed-length vector, and another to decode that vector into the target sequence. This paper established the foundation for all subsequent sequence-to-sequence models, including attention-based models and Transformers.

---

## Architecture Diagram

```text
         Source Sequence: [x₁, x₂, ..., x_T]
                      │
                      ▼
                ┌─────────────┐
                │   Encoder   │  (LSTM)
                │    LSTM     │
                └─────────────┘
                      │
                      ▼
            Context Vector (c)
                      │
                      ▼
                ┌─────────────┐
                │   Decoder   │  (LSTM)
                │    LSTM     │
                └─────────────┘
                      │
                      ▼
         Target Sequence: [y₁, y₂, ..., y_U]
```

- **Encoder:** Processes the source sequence into a context vector.
- **Decoder:** Generates the target sequence from the context vector.

---

## Key Contributions

- **Encoder-Decoder Architecture:** First practical sequence transduction system.
- **LSTM-based:** Used LSTM for both encoder and decoder.
- **Teacher Forcing:** Training uses target tokens as inputs to the decoder.
- **Reversing Source Sequence:** Improved performance for translation.

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
| `implementation_historical.py` | Forward-pass demonstration |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 1997_Hochreiter_LSTM |
| Successor (Attention) | 2014_Bahdanau_Attention |
| Successor | 2017_Transformer |