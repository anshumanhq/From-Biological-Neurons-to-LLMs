# STYLE GUIDE v1.0 – From Biological Neurons to LLMs

## 1. General Writing Philosophy
- **Voice:** Formal, objective, archival. Do not use "we believe"—use "The evidence suggests" or "It is established that".
- **Tense:** Use historical present for established facts (e.g., "Hebb *proposes*"), and past tense for specific experiments (e.g., "Rosenblatt *trained* the perceptron on...").
- **Audience:** Advanced undergraduate to early-career AI researchers. Assume the reader knows basic calculus and linear algebra.

---

## 2. Typography & Fonts (LaTeX)
- **Main Font:** `Latin Modern Roman` (default for LaTeX).
- **Math Font:** `Computer Modern` (default).
- **Monospace (Code):** `Inconsolata` or `Courier New` for any inline code snippets.
- **Base Font Size:** 10pt (for print), 11pt for digital drafts.

---

## 3. Heading Hierarchy
*(Strictly followed in the `.tex` files)*

| LaTeX Command | Level | Usage |
| :--- | :--- | :--- |
| `\chapter{...}` | 0 | Volume/Chapter title (e.g., *Chapter 1: The Brain Before AI*) |
| `\section{...}` | 1 | Major topic shift (e.g., *1.1 The Neuron Doctrine*) |
| `\subsection{...}` | 2 | Sub-topic (e.g., *1.1.1 Cajal's Morphology*) |
| `\subsubsection{...}` | 3 | Fine-grain specifics (e.g., *Synaptic Cleft Dynamics*) |
| `\paragraph{...}` | 4 | Rarely used; for bullet-style prose in dense sections. |

---

## 4. Equation Numbering & Style
- **Numbering:** Chapter-wide sequential. Format: `(Chapter.Equation)`.
  - Example: `(1.1)` for the first equation in Chapter 1.
- **LaTeX Usage:** Always use `\begin{equation} ... \end{equation}` for numbered display math.
- **Variables:** Must be italicized. Vectors bold (`\mathbf{x}`). Matrices capitalized bold (`\mathbf{W}`).
- **Reference in text:** Always write "As shown in Eq. (1.1)..." not "the above equation".

---

## 5. Figure & Table Numbering
- **Figures:** `Figure Chapter.Index` (e.g., *Figure 1.1: McCulloch-Pitts Neuron*).
- **Tables:** `Table Chapter.Index` (e.g., *Table 2.1: Perceptron Convergence Steps*).
- **Citation in caption:** Every figure **must** cite the source paper in its caption.
  - *Template:* `Figure 3.2: The Hopfield energy landscape. (Adapted from Hopfield, 1982, p. 2554)`

---

## 6. Citation Style (Author-Year)
- **In-text:** 
  - Single author: `(McCulloch, 1943)`
  - Two authors: `(Rumelhart & Hinton, 1986)`
  - Three or more: `(Vaswani et al., 2017)`
- **Narrative:** `McCulloch and Pitts (1943) demonstrated that...`
- **Multiple sources:** Sorted chronologically: `(Hebb, 1949; Rosenblatt, 1958; Minsky, 1969)`

---

## 7. Reference Style (BibTeX)
All entries must be present in the `/bibliography/` folder.
- **Journal Article:** `@article{key, author={...}, journal={...}, year={...}, doi={...}}`
- **Book:** `@book{key, author={...}, publisher={...}, year={...}, isbn={...}}`
- **Conference:** `@inproceedings{key, booktitle={...}, year={...}}`
- **Preprint:** `@misc{key, archivePrefix={arXiv}, eprint={...}, year={...}}`
- **Critical Rule:** Every `@article` **must** have a `doi` field. Scripts will validate this.

---

## 8. Timeline Style (for `/research/timeline/`)
- **Entry Format:** `YYYY-MM-DD (if known) | Event | Significance | Primary Source`
- **All dates must be cross-checked** against the original publication receipt date (not just the issue year).

---

## 9. Biography Template (for `/research/authors/`)
When documenting a pioneer, adhere to this structure:

```markdown
### [Full Name] (YYYY–YYYY)
- **Born:** Location
- **Education:** [Degrees, Years]
- **Primary Role:** [Neurophysiologist, Computer Scientist, etc.]
- **Key Contribution:** [1-line summary]
- **Key Paper:** [Title, Year, DOI]
- **Historical Context:** [1 paragraph]
```

---

## 10. Paper Analysis Template (for `/research/papers/YYYY_Author_ShortTitle/`)

Every folder must contain a `notes.md` with these exact headers:

```markdown
# [Paper Title]
- **Authors:** 
- **Year:** 
- **DOI:** 
- **Primary Claim:** 
- **Math Abstraction:** 
- **Relation to Biology:** 
- **Impact at Time of Publication:** 
- **Modern Relevance (2026 perspective):** 
- **Open Questions:** 
```

---

## 11. Code Style (Python / NumPy)
- **Imports:** Only `import numpy as np` and `import matplotlib.pyplot as plt` allowed in scratch.
- **Naming:** Functions use `snake_case`. Classes use `CamelCase`.
- **Docstrings:** Every function must have a docstring explaining the mathematical operation it implements.
- **Output:** All code must run silently (no debug prints) unless explicitly invoked with `--verbose`.

---

## 12. File Naming Conventions
- **Chapters:** `ch{XX}_{short_title}.tex` (e.g., `ch01_brain_before_ai.tex`)
- **Figures:** `fig_ch{XX}_description.png/pdf` (e.g., `fig_ch05_hopfield_energy.png`)
- **Research Papers:** `{YYYY}_{AuthorLastName}_{keyword}/`

---

*This style guide is frozen as of v0.1.0. Any future amendments require a PR to the repository and an update to CHANGELOG.md.*