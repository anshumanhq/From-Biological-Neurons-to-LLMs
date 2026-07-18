# 2023_Meta_Llama2 — Llama 2: Open Foundation and Fine-Tuned Chat Models

**Authors:** Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashmi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom  
**Year:** 2023  
**Venue:** arXiv preprint

---

## Historical Importance

Llama 2 significantly extended the LLaMA family by releasing models with up to 70B parameters, including both base models and chat‑fine‑tuned versions (Llama 2‑Chat). The paper introduced a comprehensive RLHF pipeline with safety‑specific reward models and explored the trade‑offs between helpfulness and safety. The models were released under a more permissive licence than the original LLaMA, enabling wider commercial use. Llama 2 established a new standard for open‑weight foundation models and catalysed the development of numerous derivative models.

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

- **Transformer decoder** with RMSNorm, SwiGLU.
- **RLHF alignment** with helpfulness and safety reward models.
- **Chat variants** fine‑tuned for conversation.

---

## Key Contributions

- **Open Foundation and Chat Models:** Base and instruction‑tuned variants.
- **RLHF Pipeline:** Safety‑specific reward models, careful balance of helpfulness and safety.
- **Permissive Licence:** Broader commercial use compared to LLaMA.
- **Performance:** 70B model competitive with GPT‑3.5.

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
| Predecessor | 2023_Meta_LLaMA |
| Predecessor | 2022_Ouyang_InstructGPT |
| Successor | 2023_Mistral_7B |
| Successor | 2024_Open_LLMs |