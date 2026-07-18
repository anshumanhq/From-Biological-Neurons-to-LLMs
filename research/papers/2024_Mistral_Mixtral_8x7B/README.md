# 2024_Mistral_Mixtral_8x7B — Mixtral of Experts

**Authors:** Albert Q. Jiang et al.  
**Year:** 2024  
**Venue:** arXiv preprint

---

## Historical Importance

Mixtral 8×7B introduced a sparse Mixture of Experts (MoE) architecture, extending the efficiency philosophy of Mistral 7B. The model has 8 expert networks, each with approximately 7 billion parameters, but uses only a sparse subset of experts (top‑2 routing) per token. This results in a large total parameter pool (~47B) while maintaining inference efficiency close to a dense 7B model. Mixtral demonstrated strong performance, outperforming Llama 2 70B and GPT‑3.5 on several benchmarks. The open‑weight release further solidified Mistral AI's position in the open‑weight ecosystem.

---

## Architecture Overview

```text
         Input Token
              │
              ▼
         ┌─────────────────┐
         │      Router     │  (selects top-2 experts)
         └─────────────────┘
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
   ┌──────────┐ ┌──────────┐
   │ Expert 1 │ │ Expert 2 │  (active experts)
   └──────────┘ └──────────┘
        │           │
        └─────┬─────┘
              │
              ▼
         ┌─────────────────┐
         │  Weighted Sum   │  (router-weighted)
         └─────────────────┘
```

- **Sparse Mixture of Experts (MoE):** 8 experts, top‑2 routing.
- **Total parameters:** ~47B (8 × ~7B) but sparse activation.
- **Inference efficiency:** Comparable to a 7B dense model.

---

## Key Contributions

- **Sparse MoE Architecture:** Enables large model capacity with efficient inference.
- **Top‑2 Routing:** Only two experts are active per token.
- **Open‑Weight Release:** Further expanded the open‑weight ecosystem.
- **Performance:** Outperforms larger dense models (Llama 2 70B) on many benchmarks.

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
| Predecessor | 2023_Mistral_7B |
| Predecessor | 2023_Meta_Llama2 |
| Successor | 2024_Open_LLMs |
| Successor | 2024_AgenticAI |