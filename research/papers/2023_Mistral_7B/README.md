# 2023_Mistral_7B — Mistral 7B

**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed  
**Year:** 2023  
**Venue:** arXiv preprint

---

## Historical Importance

Mistral 7B introduced a 7B-parameter efficient language model with two key architectural innovations: Grouped-Query Attention (GQA) and Sliding Window Attention (SWA). GQA reduces memory bandwidth requirements during inference, while SWA limits attention computation to local windows, enabling faster processing of long sequences. The model demonstrated strong performance on various benchmarks, often outperforming much larger models in specific evaluations. Released under an open-weight licence, Mistral 7B significantly influenced the open-weight LLM ecosystem and established efficiency as a key design goal.

---

## Architecture Overview

```text
         Input Tokens
              │
              ▼
         ┌─────────────────┐
         │  Token Embed    │
         │  + RoPE         │
         └─────────────────┘
              │
              ▼
         ┌─────────────────┐
         │  Transformer    │
         │  Decoder Layers │
         │  (32×)          │
         │                 │
         │  SWA + GQA      │
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

- **Sliding Window Attention (SWA):** Local attention window (4096 tokens).
- **Grouped-Query Attention (GQA):** Shared key/value heads for efficiency.
- **Rotary Positional Embeddings (RoPE):** Position encoding without learned embeddings.

---

## Key Contributions

- **Efficiency Innovations:** GQA (inference memory reduction), SWA (fast long-sequence processing).
- **Strong Performance:** Outperforms Llama 2 13B on many benchmarks.
- **Open-Weight Release:** Commercially permissive licence.
- **Open-Weight Ecosystem:** Significant influence on subsequent open models.

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
| Predecessor | 2023_Meta_Llama2 |
| Predecessor | 2023_Meta_LLaMA |
| Successor | 2024_Mixtral_8x7B |
| Successor | 2024_Open_LLMs |