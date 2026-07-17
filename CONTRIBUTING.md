# Contributing to "From Biological Neurons to Large Language Models"

**Project Type:** Academic Manuscript / Historical Research Archive  
**Current Phase:** Infrastructure & Drafting (v0.1.0)  
**Lead Author:** [Your Name / Lab]  

---

## 1. Introduction

Thank you for your interest in contributing. 

Unlike a software library, this repository is a **living historical record**. We do not accept random features or stylistic opinions. We accept **verifiable corrections**, **missing primary sources**, and **mathematical derivations** that enhance the archival completeness of the book.

All contributions must comply with the `STYLE_GUIDE.md` and must not break the NumPy-only constraint of the `code/` folder.

---

## 2. How to Submit Feedback (Issues)

Since this is currently a solo-author project, the preferred method for external input is via **GitHub Issues**.

### Issue Templates (Labeling)
Please label your issue correctly:

| Label | Use Case |
| :--- | :--- |
| `historical` | Incorrect date, misattributed discovery, missing pioneer. |
| `technical` | Mathematical error in equations, incorrect NumPy implementation. |
| `bibliographic` | Missing DOI, incorrect BibTeX entry, fake citation detection. |
| `style` | Typos, LaTeX rendering issues, figure labeling mismatches. |

**Required Information in an Issue:**
- **Claim being challenged:** Quote the exact sentence/equation from the manuscript.
- **Supporting Evidence:** Provide a DOI, archive link, or page number from a primary source.
- **Proposed Correction:** Suggest the exact replacement text.

---

## 3. How to Submit Corrections (Pull Requests)

PRs are welcome, but they are **strictly reviewed** for factual and stylistic compliance.

### Branch Naming Convention
```text
hotfix/correct-hebb-year
feature/add-1969-minsky-proof
docs/update-changelog
```

### Pull Request Checklist
Before submitting a PR, ensure:

- [ ] You have updated the `CHANGELOG.md` under the `[Unreleased]` section.
- [ ] Your changes pass the local citation validator: `python scripts/verify_citations.py`
- [ ] If you added a new NumPy implementation, it **does not** import TensorFlow, PyTorch, or JAX.
- [ ] If you added a new paper to `research/papers/`, you have followed the **Paper Archive Structure** (see below).
- [ ] All new LaTeX equations are correctly numbered and referenced in the text.
- [ ] You have read the `STYLE_GUIDE.md` and applied the heading hierarchy strictly.

---

## 4. Adding a New Paper to the Research Archive

If you are adding a missing primary source to `research/papers/`, you **must** create a dedicated folder following this exact pattern:

```text
research/papers/{YYYY}_{AuthorLastName}_{ShortTitle}/
├── original.pdf          # (Optional, but preferred if out of copyright)
├── notes.md              # Must follow the template defined in STYLE_GUIDE.md
├── summary.md            # 1-page abstract suitable for the book's appendix
├── equations.tex         # All mathematical expressions from the paper, copied verbatim
├── bibliography.bib      # BibTeX entry for this paper ONLY (for local reference)
├── implementation.py     # NumPy scratch re-implementation (if applicable)
├── diagrams/             # Hand-drawn or re-plotted figures from the paper
├── timeline.md           # Where this paper sits in the global 1943–2026 timeline
└── questions.md          # Open questions or unresolved debates surrounding this paper
```

**Failure to include at least `notes.md` and `bibliography.bib` will result in PR rejection.**

---

## 5. Code of Conduct (Academic Civility)

This project adheres to a simple, strict policy:

- **No ad hominem attacks** on historical figures (e.g., Minsky vs. Rosenblatt debates must be factual, not emotional).
- **No hype language.** Words like "revolutionary", "game-changer", or "magical" are banned from the manuscript unless directly quoted from a primary source.
- **Constructive criticism only.** If you find a flaw in the math, provide the correct derivation, not just a complaint.

---

## 6. Legal & Licensing

- All prose contributions fall under the **CC BY-NC-ND 4.0** license.
- Code contributions (NumPy) fall under the **MIT** license for maximum educational reusability.
- By submitting a PR, you agree to these terms.

---

## 7. Getting Help

For questions about the folder structure or compilation:

- Read the `README.md` first.
- Check the `docs/` folder for supplementary notes.
- If all else fails, open a `question` issue (but expect a delayed response, as this is a part-time research project).

---

**Last Updated:** 2026-07-17