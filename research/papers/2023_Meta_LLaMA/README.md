# 2023_Meta_LLaMA — LLaMA: Open and Efficient Foundation Language Models

**Authors:** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample  
**Year:** 2023  
**Venue:** arXiv preprint

---

## Historical Importance

LLaMA demonstrated that highly capable foundation language models could be trained at comparatively smaller scales and released as research models, catalysing the open‑weight LLM ecosystem. It achieved competitive performance with GPT‑3 using publicly available data and opened the door for a wave of derivative models (Alpaca, Vicuna, Llama‑2, Mistral, etc.), shaping the modern open‑weight landscape.

---

## Architecture Overview

```text
         Input Tokens
              │
              ▼
         ┌─────────────────┐
         │  Token Embed    │
         │  + Positional   │
         └─────────────────┘
              │
              ▼
         ┌─────────────────┐
         │  Transformer    │
         │  Decoder Layers │
         │  (N×)           │
         │                 │
         │  Causal Self-   │
         │  Attention      │
         │  + SwiGLU       │
         │  + RMSNorm      │
         └─────────────────┘
              │
              ▼
         ┌─────────────────┐
         │  Linear +       │
         │  Softmax        │
         └─────────────────┘
              │
              ▼
         Output Tokens
```

- **Transformer decoder** architecture with efficiency improvements.
- **RMSNorm** instead of LayerNorm.
- **SwiGLU** activation.
- **Open‑weight** release under research licence.

---

## Key Contributions

- **Efficient Training:** Competitive performance at smaller scales (7B–65B).
- **Public Data:** All training data is publicly available.
- **Open-Weight Release:** Catalysed the open‑source LLM ecosystem.
- **Benchmark Performance:** LLaMA‑13B outperformed GPT‑3 on several tasks.

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
| `implementation_historical.py` | Conceptual demonstration |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2023_OpenAI_GPT4 |
| Predecessor | 2020_Brown_GPT3 |
| Successor | 2023_Meta_Llama2 |
| Successor | 2023_Mistral_7B |