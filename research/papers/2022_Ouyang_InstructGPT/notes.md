# Training Language Models to Follow Instructions with Human Feedback

- **Paper ID:** `2022_Ouyang_InstructGPT`
- **Authors:** Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
- **Year:** 2022
- **Venue / Journal:** arXiv preprint
- **DOI:** 10.48550/arXiv.2203.02155
- **Primary Subject:** Natural Language Processing / Reinforcement Learning / Alignment

---

## 1. Historical Background

By 2022, GPT‑3 had demonstrated impressive few‑shot capabilities (175B parameters). However, the base GPT‑3 model did not reliably follow user instructions, could generate harmful or unhelpful content, and often failed to align with human preferences. Ouyang et al. proposed a fine‑tuning pipeline using **Reinforcement Learning from Human Feedback (RLHF)** to address these issues, building on earlier work in alignment and preference learning.

---

## 2. Problem Statement

The authors addressed the problem of **misalignment** between base language models and human preferences. They asked: can we fine‑tune a pretrained language model using human feedback to make it more helpful, truthful, and harmless? The goal was to create a model that reliably follows instructions and aligns with human values.

---

## 3. Primary Claim

The paper's central claim is that **a 1.3B parameter InstructGPT model, fine‑tuned with RLHF, is preferred by human evaluators over the 175B GPT‑3 base model** on the evaluated prompt distribution. This demonstrates that alignment through human feedback can compensate for model size. The paper also reports improvements in truthfulness and reductions in toxic output generation.

---

## 4. Math Abstraction

**Supervised Fine‑Tuning (SFT) Objective:**

```latex
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \log \pi_{\theta}(y \mid x)
```

**Reward Model (RM) Objective (Pairwise Ranking):**

```latex
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{RM}}}
\log \sigma\left( r_{\phi}(x, y_w) - r_{\phi}(x, y_l) \right)
```

**PPO (RLHF) Objective (with KL penalty):**

```latex
\mathcal{L}_{\text{RL}} = -\mathbb{E}_{x \sim \mathcal{D}_{\text{RL}}, y \sim \pi_{\theta}(x)}
\left[ r_{\phi}(x, y) - \beta \cdot \text{KL}\left( \pi_{\theta} \parallel \pi_{\text{ref}} \right) \right]
```

**Overall Pipeline:**

```latex
\pi_{\text{GPT-3}} \xrightarrow{\text{SFT}} \pi_{\text{SFT}} \xrightarrow{\text{RM}} r_{\phi} \xrightarrow{\text{PPO/RLHF}} \pi_{\text{InstructGPT}}
```

---

## 5. Relation to Biology

InstructGPT is **not** biologically inspired. The RLHF framework is derived from reinforcement learning and preference learning, not neuroscience or cognitive science.

---

## 6. Original Paper Analysis

The paper introduced several innovations:

1. **SFT → RM → PPO Pipeline:** A practical RLHF pipeline for LLM alignment.
2. **Human Preference Data:** Collected from labelers for SFT and reward modeling.
3. **Alignment vs Capability:** 1.3B InstructGPT preferred over 175B GPT‑3.
4. **Improvements:** Better instruction following, truthfulness, and reduced toxicity.
5. **Limitations:** Reward hacking, overoptimisation, distribution shift, and alignment tax.

**Important:** InstructGPT did not make the model fundamentally more knowledgeable through RLHF. The major goal was to make pretrained capabilities more reliably follow human instructions and preferences.

---

## 7. Algorithm / Method

**Training Procedure:**

1. **Supervised Fine‑Tuning (SFT):**
   - Collect human‑written demonstrations of instruction‑response pairs.
   - Fine‑tune the pretrained model on these demonstrations.
   - Model learns to mimic human responses.

2. **Reward Model (RM) Training:**
   - Collect pairwise human preferences (response A vs B).
   - Train a reward model to predict which response is preferred.
   - Model learns to assign scores to responses.

3. **PPO / RLHF:**
   - Use the reward model to score model‑generated responses.
   - Optimise the language model to generate highly rewarded responses.
   - Apply KL penalty to prevent the model from drifting too far from the reference model.

4. **Evaluation:**
   - Human evaluators compare model outputs.
   - Metrics: helpfulness, truthfulness, harmlessness.

---

## 8. NumPy Scratch Implementation

```python
import numpy as np

class RewardModel:
    def __init__(self):
        self.weights = np.random.randn(1, 1) * 0.1

    def predict(self, prompt, response):
        return float(np.dot(np.random.randn(1, 1), np.random.randn(1, 1)))

class PPOOptimizer:
    def __init__(self, model, reward_model, ref_model, kl_penalty=0.01):
        self.model = model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.kl_penalty = kl_penalty

    def optimize(self, prompts, epochs=10):
        # Objective: maximize reward - β * KL(π || π_ref)
        print("Optimizing against reward model with KL penalty.")
        return self.model
```

The implementation demonstrates the RLHF pipeline.

---

## 9. Limitations (As Acknowledged by the Authors)

- **Reward Hacking:** The model may exploit the reward model rather than improving.
- **Overoptimisation:** Maximising reward can degrade helpfulness.
- **Distribution Shift:** Performance may not generalise outside the training distribution.
- **Alignment Tax:** Alignment can reduce performance on some academic NLP benchmarks.
- **Data Quality:** Depends heavily on human labeler quality and consistency.

---

## 10. Impact at Time of Publication

The paper established RLHF as the standard method for aligning LLMs with human preferences. It demonstrated that alignment can significantly improve model behavior on instruction‑following tasks, even with smaller models (1.3B vs 175B). The work directly influenced ChatGPT, which was released shortly after as a conversational interface to InstructGPT.

---

## 11. Influence on Later Research

- **ChatGPT (2022):** Built on InstructGPT with a conversational interface.
- **GPT‑4 (2023):** Used RLHF for alignment.
- **Claude (2023):** Used constitutional AI and RLHF.
- **RLHF Scaling:** Many subsequent LLMs adopted RLHF for alignment.

---

## 12. Modern Relevance (2026 Perspective)

InstructGPT's RLHF pipeline remains the dominant method for aligning LLMs. The core ideas—SFT, reward modeling, and PPO—are still used in GPT‑4, Claude, Gemini, and other state‑of‑the‑art models. The paper's insights about alignment vs capability, reward hacking, and distribution shift continue to shape AI safety research.

---

## 13. Primary Source Paraphrase

- RLHF fine‑tuning improves instruction following and alignment.
- 1.3B InstructGPT is preferred over 175B GPT‑3.
- Improvements in truthfulness and reduced toxicity are reported.
- Limitations include reward hacking and distribution shift.
- RLHF aligns pretrained capabilities with human preferences.

---

## 14. Historical Timeline

- **Before:**
  - 2017: Transformer
  - 2018: GPT‑1
  - 2019: GPT‑2
  - 2020: GPT‑3
- **Publication:**
  - 2022: InstructGPT
- **After:**
  - 2022: ChatGPT
  - 2023: GPT‑4

---

## 15. Common Misconceptions

- **Misconception 1:** "RLHF was introduced in InstructGPT."
  - **Fact:** RLHF existed before InstructGPT; the paper demonstrated a practical, influential pipeline.
- **Misconception 2:** "InstructGPT made the model more knowledgeable."
  - **Fact:** RLHF made pretrained capabilities more reliable, not more knowledgeable.
- **Misconception 3:** "InstructGPT eliminated all harmful outputs."
  - **Fact:** It reduced toxicity but did not eliminate it.

---

## 16. Implementation Verification

```python
def test_instructgpt_pipeline():
    instructgpt = InstructGPT_2022("GPT-3-175B")
    demonstrations = [("Prompt", "Response")]
    preference_data = [("Prompt", "A", "B")]
    prompts = ["Prompt"]
    instructgpt.sft(demonstrations)
    rm = instructgpt.reward_modeling(preference_data)
    instructgpt.ppo(prompts, rm, None)
    print("InstructGPT pipeline demonstration successful.")
```

---

## 17. Cross References

- **Predecessor:** 2020_Brown_GPT3
- **Predecessor:** 2019_Radford_GPT2
- **Successor:** 2022_OpenAI_ChatGPT
- **Successor:** 2023_OpenAI_GPT4

---

## 18. Open Questions

1. How can reward hacking be mitigated?
2. What is the optimal balance between alignment and capability?
3. How does distribution shift affect RLHF generalisation?
4. Can RLHF be made more sample‑efficient?
5. How do different KL penalty strengths affect alignment?
