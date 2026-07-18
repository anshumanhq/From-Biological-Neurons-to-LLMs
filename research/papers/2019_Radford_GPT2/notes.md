# Language Models are Unsupervised Multitask Learners

- **Paper ID:** `2019_Radford_GPT2`
- **Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- **Year:** 2019
- **Venue / Journal:** OpenAI Technical Report
- **Primary Subject:** Natural Language Processing / Generative Pre-training / Zero-shot Learning

---

## 1. Historical Background

By 2019, GPT‑1 (2018) had established the "pre‑train then fine‑tune" paradigm. However, the approach still required task‑specific labelled data for fine‑tuning. Radford et al. asked: can a language model learn to perform tasks without any fine‑tuning? They hypothesised that scaling the model and training it on a large, diverse corpus might enable the model to perform tasks via zero‑shot transfer, where the task is specified by the prompt alone.

---

## 2. Problem Statement

The authors addressed the problem of **task‑specific data requirements** in NLP. They asked: can a large language model perform new tasks without supervised fine‑tuning, simply by conditioning on a prompt? The goal was to demonstrate that language modelling is a general‑purpose learning framework that can solve tasks without explicit supervision.

---

## 3. Primary Claim

The paper's central claim is that **scaling a language model** to 1.5B parameters and training it on a large, diverse web corpus (WebText, 40GB) enables **zero‑shot task transfer** where the model performs tasks without any task‑specific fine‑tuning. The model achieves competitive performance on tasks like translation, question answering, and summarisation solely through conditioning on the prompt.

---

## 4. Math Abstraction

**Autoregressive Language Modelling Objective:**

```latex
\mathcal{L} = -\sum_{i} \log P(u_i \mid u_{<i}; \Theta)
```

**Transformer Decoder Layer (with causal masking):**

```latex
h_l = \text{block}(h_{l-1}) \quad \forall l \in [1, L]
```

**Causal Masked Self-Attention:**

```latex
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
```

**Byte-level BPE Tokenization:**

```latex
\text{ByteBPE}(x) = \text{merge}(\text{bytes}(x), \text{merges})
```

**Zero-shot Task Transfer:**

```latex
P(\text{answer} \mid \text{context}) = \text{LM}(\text{prompt})
```

---

## 5. Relation to Biology

GPT‑2 is **not** biologically inspired. It is a purely computational architecture based on the Transformer decoder.

---

## 6. Original Paper Analysis

The paper introduced several innovations:

1. **Zero-shot Task Transfer:** The model performs tasks without supervised fine‑tuning.
2. **WebText Dataset:** 40GB of high‑quality web pages curated from Reddit (outbound links with ≥3 karma).
3. **Byte‑level BPE:** Tokenization at the byte level, handling any text without character‑level limitations.
4. **Scale:** Four model sizes: Small (117M), Medium (345M), Large (774M), XL (1.5B).
5. **Staged Release:** Concerns about potential misuse led to a staged release over several months.

**Important:** GPT‑2 did not "invent" zero‑shot learning. Its contribution was demonstrating that scaling a language model and training on a large, diverse web corpus could produce strong zero‑shot task performance without task‑specific fine‑tuning.

---

## 7. Algorithm / Method

**Training Procedure:**

1. **Pre‑training Phase:** Train on WebText (40GB of web pages) using autoregressive language modelling.
2. **Task Transfer (Zero-shot):** Condition the model on a prompt describing the task (e.g., "Translate English to French: cat → ") and generate the output.
3. **No Fine‑tuning:** The model is not fine‑tuned for specific tasks; all tasks are performed in a zero‑shot setting.

**Architecture:** GPT‑2 uses the same decoder‑only Transformer architecture as GPT‑1, but with more layers, larger hidden dimensions, and more attention heads, scaled to 1.5B parameters for the XL variant.

---

## 8. NumPy Scratch Implementation

```python
import numpy as np

class CausalMask:
    @staticmethod
    def create(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

class ScaledDotProductAttentionWithMask:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        return np.dot(attn_weights, V), attn_weights
```

The implementation above demonstrates causal masking and decoder‑only attention.

---

## 9. Limitations (As Acknowledged by the Authors)

- **Computational Cost:** Training the 1.5B model required significant computational resources.
- **Data Dependence:** Performance scales with the size and quality of the pre‑training corpus.
- **Zero‑shot Limitations:** Performance lags behind fine‑tuned models on many tasks.
- **Tokenisation:** Byte‑level BPE can produce suboptimal segmentations for some languages.
- **Prompt Sensitivity:** Zero‑shot performance depends heavily on prompt formulation.

---

## 10. Impact at Time of Publication

The paper demonstrated that scaling a language model to 1.5B parameters and training on a large, diverse web corpus could produce strong zero‑shot task performance. It established language modelling as a general‑purpose learning framework and introduced the concept of zero‑shot task transfer. The staged release attracted significant attention and sparked discussions about the responsible use of AI.

---

## 11. Influence on Later Research

- **GPT‑3 (2020):** Scaled‑up version (175B parameters) with few‑shot learning.
- **InstructGPT (2022):** Fine‑tuned with RLHF for instruction following.
- **ChatGPT (2022):** Conversational AI built on GPT‑3.5.
- **GPT‑4 (2023):** Multimodal reasoning.
- **Zero‑shot and Few‑shot Learning:** Established as a standard evaluation paradigm.

---

## 12. Modern Relevance (2026 Perspective)

GPT‑2 is the direct precursor to modern LLMs. Its core ideas—scaling, zero‑shot transfer, and language modelling as a general‑purpose framework—remain central to GPT‑4, Claude, Gemini, and other state‑of‑the‑art models. The staged release and discussions about potential misuse also influenced AI safety and ethics.

---

## 13. Primary Source Paraphrase

- Language models can perform tasks without supervised fine‑tuning.
- Scaling the model and training on a diverse corpus enables zero‑shot transfer.
- Byte‑level BPE tokenization handles any text.
- The model family includes four sizes up to 1.5B parameters.
- The staged release reflected concerns about potential misuse.

---

## 14. Historical Timeline

- **Before:**
  - 2017: Transformer
  - 2018: GPT‑1
- **Publication:**
  - 2019: GPT‑2 (OpenAI Technical Report)
- **After:**
  - 2020: GPT‑3
  - 2022: InstructGPT
  - 2022: ChatGPT
  - 2023: GPT‑4

---

## 15. Common Misconceptions

- **Misconception 1:** "GPT‑2 invented zero‑shot learning."
  - **Fact:** Zero‑shot learning existed before; GPT‑2 demonstrated its scalability.
- **Misconception 2:** "GPT‑2 was the first model to use byte‑level BPE."
  - **Fact:** Byte‑level BPE was introduced earlier; GPT‑2 popularised it.
- **Misconception 3:** "GPT‑2 was fine‑tuned on specific tasks."
  - **Fact:** GPT‑2 was not fine‑tuned; it performed tasks in a zero‑shot setting.

---

## 16. Implementation Verification

```python
def test_gpt2_forward():
    gpt2 = GPT2_2019(model_size="xl", d_model=1600, num_heads=25, num_layers=48, d_ff=6400)
    x = np.random.randn(1, 10, 1600)
    out = gpt2.forward(x)
    assert out.shape == (1, 10, 1600)
    print("GPT-2 forward pass with causal masking successful.")
```

---

## 17. Cross References

- **Predecessor:** 2018_Radford_GPT
- **Predecessor:** 2017_Vaswani_Transformer
- **Successor:** 2020_Brown_GPT3
- **Successor:** 2022_Ouyang_InstructGPT

---

## 18. Open Questions

1. How does the scale of the pre‑training corpus affect zero‑shot performance?
2. What is the optimal trade‑off between model size and training data?
3. How can zero‑shot transfer be improved without fine‑tuning?
4. What are the limits of zero‑shot learning in language models?
5. How does byte‑level BPE affect cross‑lingual performance?
