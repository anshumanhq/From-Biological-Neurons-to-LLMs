# 2019_Radford_GPT2 — Language Models are Unsupervised Multitask Learners

**Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever  
**Year:** 2019  
**Venue:** OpenAI Technical Report

---

## Historical Importance

GPT-2 demonstrated that scaling a Transformer language model to 1.5 billion parameters and training on a large, diverse web corpus (WebText, 40GB) could produce strong zero-shot task performance without task-specific fine-tuning. It established language modelling as a general-purpose learning framework and introduced byte-level BPE tokenization.

---

## Architecture Overview

```text
         Input Tokens (byte-level BPE)
               │
               ▼
         ┌─────────────┐
         │ Token Embed │
         │ + Position  │
         └─────────────┘
               │
               ▼
         ┌─────────────┐
         │   48×       │
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
- **Byte-level BPE** tokenization handles any text.
- **Zero-shot transfer** without task-specific fine-tuning.

---

## Model Variants

| Variant | Parameters | Layers | d_model | Heads |
| :--- | :--- | :--- | :--- | :--- |
| Small | 117M | 12 | 768 | 12 |
| Medium | 345M | 24 | 1024 | 16 |
| Large | 774M | 36 | 1280 | 20 |
| XL | 1.5B | 48 | 1600 | 25 |

---

## Key Contributions

- **Zero-shot Task Transfer:** Language models can perform tasks without fine-tuning.
- **WebText Dataset:** 40GB of high-quality web pages.
- **Byte-level BPE:** Tokenization without character-level limitations.
- **Scale:** 1.5B parameters (largest version).
- **Staged Release:** Concerns about misuse led to gradual release.

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
| Predecessor | 2018_Radford_GPT |
| Predecessor | 2017_Vaswani_Transformer |
| Successor | 2020_Brown_GPT3 |
| Successor | 2022_Ouyang_InstructGPT |