# 2017_Vaswani_Transformer — Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Year:** 2017  
**Venue:** NeurIPS (Advances in Neural Information Processing Systems)

---

## Historical Importance

The Transformer introduced a sequence transduction architecture based solely on attention mechanisms, eliminating recurrence and convolution. The scaled dot-product attention \( \text{softmax}(QK^T/\sqrt{d_k})V \) and multi-head attention allow parallel processing of sequences, enabling efficient training and state-of-the-art results on machine translation. This architecture became the foundation for all modern large language models.

---

## Architecture Diagram

```text
         Input Embedding
               │
               ▼
         ┌─────────────┐
         │ Positional  │
         │  Encoding   │
         └─────────────┘
               │
               ▼
         ┌─────────────┐
         │   Encoder   │  (N× layers)
         │             │
         │  Self-Attn  │
         │   + FFN     │
         └─────────────┘
               │
               ▼
         ┌─────────────┐
         │   Decoder   │  (N× layers)
         │             │
         │  Masked     │
         │  Self-Attn  │
         │  Cross-Attn │
         │   + FFN     │
         └─────────────┘
               │
               ▼
         Output Embedding
```

- **Scaled Dot-Product Attention:** \( \text{softmax}(QK^T/\sqrt{d_k})V \)
- **Multi-Head Attention:** Concatenates multiple attention heads.
- **Positional Encoding:** Sin/cos positional embeddings.
- **Encoder:** Self-attention + position-wise FFN (residual + layer norm).
- **Decoder:** Masked self-attention + cross-attention + FFN.

---

## Key Contributions

- **Scaled Dot-Product Attention:** Efficient and parallelisable.
- **Multi-Head Attention:** Captures different types of relationships.
- **Positional Encoding:** Provides position information without recurrence.
- **Encoder-Decoder Architecture:** No recurrence or convolution.
- **State-of-the-Art Translation:** WMT 2014 English-German and English-French.

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
| `implementation_historical.py` | Forward‑pass demonstration |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2014_Sutskever_Seq2Seq |
| Predecessor | 2014_Bahdanau_Attention |
| Successor | 2018_GPT |
| Successor | 2018_BERT |
| Successor | 2020_GPT3 |