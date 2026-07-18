# 2012_Krizhevsky_AlexNet — ImageNet Classification with Deep Convolutional Neural Networks

**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Year:** 2012  
**Venue:** NeurIPS (Advances in Neural Information Processing Systems)

---

## Historical Importance

AlexNet marked the beginning of the **modern deep-learning era**. It demonstrated that a deep CNN with **ReLU activations**, **dropout**, **data augmentation**, and **GPU training** could achieve breakthrough performance on the ImageNet challenge (ILSVRC 2012), reducing the top‑5 error rate from 26% to 15.3%.

---

## Key Contributions

- **ReLU Activation:** Faster training than tanh/sigmoid.
- **Dropout:** Prevents overfitting in fully‑connected layers.
- **Data Augmentation:** Image translations, horizontal reflections.
- **GPU Training:** Two‑GPU model parallelism.
- **Max Pooling:** Used instead of average pooling.
- **Large‑Scale Training:** ImageNet (1.2M images, 1000 classes).

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
| Predecessor | 1998_LeCun_LeNet5 |
| Predecessor | 1989_LeCun_CNN |
| Successor | 2015_ResNet |
| Successor | 2017_Transformer |