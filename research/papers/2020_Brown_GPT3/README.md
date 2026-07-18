# 2020_Brown_GPT3 — Language Models are Few-Shot Learners

**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei  
**Year:** 2020  
**Venue:** NeurIPS (Advances in Neural Information Processing Systems)

---

## Historical Importance

GPT‑3 scaled the autoregressive Transformer language model to 175 billion parameters, demonstrating that scaling substantially improves in‑context learning, particularly in the few‑shot setting. The model was evaluated in zero‑shot, one‑shot, and few‑shot settings without gradient updates or task‑specific fine‑tuning. The paper established scaling as the central experimental variable and documented both successes and limitations.

---

## Architecture Overview

```text
         Input Tokens + Prompt Demonstrations
                      │
                      ▼
         ┌─────────────────────────┐
         │       96×               │
         │   Transformer Decoder   │
         │                         │
         │  Masked Self-Attention  │
         │  + Position-wise FFN    │
         └─────────────────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │      Linear + Softmax   │
         └─────────────────────────┘
                      │
                      ▼
         Output Token Probabilities
```

- **175B parameters**, 96 layers, 12,288 d_model, 96 heads.
- **Zero-shot, one-shot, few-shot** evaluation.
- **No gradient updates** during evaluation.
- **In-context learning** via prompting.

---

## Model Variants

| Variant | Parameters | Layers | d_model | Heads |
| :--- | :--- | :--- | :--- | :--- |
| Small | 125M | 12 | 768 | 12 |
| Medium | 350M | 24 | 1024 | 16 |
| Large | 760M | 24 | 1536 | 16 |
| XL | 1.3B | 24 | 2048 | 24 |
| 2.7B | 2.7B | 32 | 2560 | 32 |
| 6.7B | 6.7B | 32 | 4096 | 32 |
| 13B | 13B | 40 | 5140 | 40 |
| 175B | 175B | 96 | 12288 | 96 |

---

## Key Contributions

- **175B Parameters:** Largest language model at the time.
- **In-context Learning:** Zero-shot, one-shot, few-shot without fine‑tuning.
- **Scaling:** Performance improves substantially with scale.
- **Web-scale Training:** Diverse web text, books, and Wikipedia.
- **Benchmark Contamination:** Documented limitations of large web corpora.

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
| `implementation_historical.py` | Forward‑pass with in‑context learning |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2019_Radford_GPT2 |
| Predecessor | 2018_Radford_GPT |
| Successor | 2022_Ouyang_InstructGPT |
| Successor | 2022_OpenAI_ChatGPT |