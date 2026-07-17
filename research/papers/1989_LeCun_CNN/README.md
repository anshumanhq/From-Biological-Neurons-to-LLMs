# 1989_LeCun_CNN — Backpropagation Applied to Handwritten Zip Code Recognition

**Authors:** Yann LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel  
**Year:** 1989  
**DOI:** 10.1162/neco.1989.1.4.541  

---

## Historical Importance

This is the first major paper to apply **backpropagation** to a **convolutional architecture** for handwritten digit recognition. It established the blueprint for all modern CNNs: local receptive fields, weight sharing, and hierarchical feature extraction.

The paper directly builds on:
- **Fukushima (1980) — Neocognitron:** Architectural inspiration (local receptive fields, pooling).
- **Rumelhart, Hinton & Williams (1986) — Backpropagation:** The learning algorithm.

---

## Available Files

| File | Description |
| :--- | :--- |
| `notes.md` | Full 18‑section historical analysis. |
| `summary.md` | One‑page abstract. |
| `equations.tex` | Core LaTeX equations. |
| `bibliography.bib` | BibTeX entry. |
| `timeline.md` | Historical timeline context. |
| `questions.md` | Open questions and debates. |
| `metadata.yaml` | Structured metadata for automation. |
| `reproducibility.md` | Experimental reproducibility notes. |
| `README.md` | This file. |

---

## Implementations

| File | Description |
| :--- | :--- |
| `implementation_historical.py` | Historically faithful: sigmoid activation, average pooling, SGD, uniform initialisation. |
| `implementation_modern.py` | Modern pedagogical: ReLU, max pooling, Adam/He initialisation. |

Both implementations are in pure NumPy with no external dependencies.

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor (Architecture) | 1980_Fukushima_Neocognitron |
| Predecessor (Learning) | 1986_Rumelhart_Hinton_Williams_Backprop |
| Successor | 1998_LeNet-5 |
| Successor | 2012_AlexNet |

---

## Reproducibility

See `reproducibility.md` for dataset, hardware, and experimental setup details.