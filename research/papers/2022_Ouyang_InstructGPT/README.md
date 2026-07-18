# 2022_Ouyang_InstructGPT — Training Language Models to Follow Instructions with Human Feedback

**Authors:** Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe  
**Year:** 2022  
**Venue:** arXiv preprint

---

## Historical Importance

InstructGPT demonstrated that fine-tuning a pretrained language model with **Reinforcement Learning from Human Feedback (RLHF)** could significantly improve instruction following and alignment with human preferences. The pipeline involved supervised fine-tuning on human-written demonstrations, reward model training on human preference rankings, and PPO-based RLHF optimization. Human evaluators preferred outputs from the 1.3B InstructGPT model over the 175B GPT-3 base model on the evaluated prompt distribution.

---

## RLHF Pipeline Diagram

```text
         Pretrained GPT-3 (175B)
                    │
                    ▼
         ┌─────────────────────────┐
         │ Supervised Fine-Tuning  │
         │ (SFT) on demonstrations │
         └─────────────────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │ Reward Model (RM)       │
         │ Training on human       │
         │ preference rankings     │
         └─────────────────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │ PPO / RLHF              │
         │ Optimize against        │
         │ reward model            │
         │ + KL penalty            │
         └─────────────────────────┘
                    │
                    ▼
         InstructGPT
         (better instruction following)
```

- **SFT:** Model learns to mimic human-written responses.
- **Reward Model:** Learns to assign scores to responses based on human preferences.
- **PPO / RLHF:** Optimises the model to generate highly rewarded responses.

---

## Key Contributions

- **SFT → RM → PPO Pipeline:** Practical RLHF for aligning LLMs.
- **Alignment vs Capability:** 1.3B InstructGPT preferred over 175B GPT-3.
- **Improvements:** Better instruction following, truthfulness, and reduced toxicity.
- **Reward Hacking:** Documented limitations of RLHF (overoptimisation).
- **Distribution Shift:** Performance generalisation across tasks.

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
| `implementation_historical.py` | RLHF pipeline demonstration |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2020_Brown_GPT3 |
| Predecessor | 2019_Radford_GPT2 |
| Successor | 2022_OpenAI_ChatGPT |
| Successor | 2023_OpenAI_GPT4 |