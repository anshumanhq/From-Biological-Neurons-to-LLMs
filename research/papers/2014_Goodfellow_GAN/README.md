# 2014_Goodfellow_GAN — Generative Adversarial Nets

**Authors:** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
**Year:** 2014  
**Venue:** NeurIPS (Advances in Neural Information Processing Systems)

---

## Historical Importance

GANs introduced **adversarial training** as a general framework for learning generative models through competition between a generator and discriminator. The minimax objective connects the training dynamics to Jensen-Shannon divergence minimization. This paper established a new paradigm for generative modelling that has since been extended to image, text, and audio generation.

---

## Architecture Diagram

```text
         Random Noise (z)
               │
               ▼
         ┌───────────┐
         │ Generator │  (MLP)
         │    (G)    │
         └───────────┘
               │
               ▼
         Generated Sample (G(z))
               │
               ├─────────────────────┐
               │                     │
               ▼                     ▼
         ┌───────────┐         ┌───────────┐
         │ Real Data │         │  G(z)     │
         │ (x ~ p_data)        │ (fake)    │
         └───────────┘         └───────────┘
               │                     │
               └──────────┬──────────┘
                          │
                          ▼
                   ┌───────────┐
                   │Discriminator│
                   │    (D)    │
                   └───────────┘
                          │
                          ▼
                   Real / Fake?
```

- **Generator (G):** Maps latent noise to generated samples.
- **Discriminator (D):** Classifies samples as real or fake.
- **Training:** Alternating updates to G and D in a minimax game.

---

## Key Contributions

- **Adversarial Training:** Generators and discriminators learn through competition.
- **Minimax Objective:** \( \min_G \max_D V(D,G) \) with JS divergence interpretation.
- **MLP-based:** Original GAN used MLPs (CNNs came later with DCGAN).
- **Theoretical Foundation:** Proof of convergence under ideal conditions.

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
| `implementation_historical.py` | Forward-pass demonstration |
| `implementation_modern.py` | Modern conceptual implementation |

---

## Cross-References

| Relationship | Paper |
| :--- | :--- |
| Predecessor | 2012_Krizhevsky_AlexNet |
| Predecessor | 2006_Hinton_DBN |
| Successor (Image GANs) | 2015_DCGAN |
| Successor (Stable Training) | 2017_WGAN |
| Successor | 2018_StyleGAN |