# 2018_Radford_GPT — Improving Language Understanding by Generative Pre-Training

**Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever  
**Year:** 2018  
**Venue:** OpenAI Technical Report

---

## Historical Importance

GPT (2018) introduced generative pre-training of a Transformer decoder on a large text corpus (BooksCorpus), followed by supervised fine-tuning on downstream tasks. This established the "pre-train then fine-tune" paradigm that became the foundation for modern LLMs. The model uses a 12-layer Transformer decoder with masked self-attention (causal masking) to perform autoregressive language modelling.

---

## Architecture Diagram

```text
         Input Tokens
               │
               ▼
         ┌─────────────┐
         │ Token Embed │
         │ + Position  │
         └─────────────┘
               │
               ▼
         ┌─────────────┐
         │   12×       │
         │ Decoder     │
         │ Blocks      │
         │             │
         │ Masked      │
         │ Self-Attn  │
         │ + FFN       │
         └─────────────┘
               │
               ▼
         ┌─────────────┐
         │   Linear    │
         │   + Softmax │
         └─────────────┘
               │
               ▼
         Output Token Probabilities
```

- **Decoder-only architecture** (no encoder).
- **Causal masking** prevents attending to future tokens.
- **Autoregressive generation** left-to-right.

---

## Key Contributions

- **Generative Pre-training:** Unsupervised learning on BooksCorpus (5GB text).
- **Supervised Fine-tuning:** Transfer learning to downstream tasks.
- **Decoder-only Transformer:** Simplified architecture compared to full Transformer.
- **117M Parameters:** 12 layers, 768 d_model, 12 heads.
- **State-of-the-art results:** Outperformed previous methods on multiple NLP tasks.

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
| `implementation_historical.py` | Forward‑pass with causal masking |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2017_Vaswani_Transformer |
| Successor | 2019_Radford_GPT2 |
| Successor | 2020_Brown_GPT3 |