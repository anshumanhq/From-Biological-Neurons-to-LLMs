# 2015_He_ResNet — Deep Residual Learning for Image Recognition

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Year:** 2015 (arXiv) / 2016 (CVPR)  
**Venue:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2016

---

## Historical Importance

ResNet introduced residual learning with identity shortcut connections to address the degradation problem in very deep networks. It won the ILSVRC 2015 competition with 3.57% top‑5 error, surpassing human‑level performance for the first time. Residual connections have since become a fundamental component of many architectures, including Transformers.

---

## Architecture Diagram

```text
         Input: x
            │
            ▼
      ┌─────────────┐
      │   Conv / BN  │
      │   ReLU       │
      │   Conv / BN  │
      └─────────────┘
            │
            ▼
      F(x) ─┼─────── x ──────┐
            │                │
            └────────────────┘
                    │
                    ▼
              ReLU (F(x) + x)
                    │
                    ▼
              Output: y
```

- **Residual Block:** \( \mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x} \)
- **Identity Shortcut:** Adds no parameters.
- **Bottleneck Block:** Efficient for deeper networks.

---

## Key Contributions

- **Residual Learning:** \( \mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x} \)
- **Identity Shortcuts:** Direct connections skipping layers.
- **ILSVRC 2015:** 3.57% top‑5 error (first to beat human performance).
- **Depth:** 34‑layer, 50‑layer, 101‑layer, 152‑layer variants.

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
| `implementation_historical.py` | Forward‑pass demonstration |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2012_Krizhevsky_AlexNet |
| Predecessor | 1998_LeCun_LeNet5 |
| Successor | 2017_Transformer |
| Successor | 2018_GPT |