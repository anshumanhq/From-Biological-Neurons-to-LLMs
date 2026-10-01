# From Biological Neurons to Large Language Models

**Version:** 0.6.0 (Modern LLMs & Open-Weight Era)  
**License:** CC BY-NC-ND 4.0

---

## Status Badges

[![Validation](https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs/actions/workflows/validate.yml/badge.svg)](https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs/actions/workflows/validate.yml)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Papers Archived](https://img.shields.io/badge/Papers%20Archived-31-brightgreen)](research/index.yaml)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)

---

## Book Objective

This project is a comprehensive, historically accurate, and technically rigorous chronicle of the evolution of Artificial Intelligence. It traces the mathematical and biological lineage from the McCulloch-Pitts neuron (1943) to modern Transformer-based Large Language Models (2026).

Unlike typical history books, this repository treats the book as a **living research archive**. Every claim is tied to a primary source, every equation is re-implemented from scratch in NumPy, and every chapter is version-controlled.

---

## Project Status Definitions

This repository distinguishes four research states:

### Archived

The paper exists in the repository with its core metadata, source information, notes, and associated research files.

### Audited

The archived material has undergone a structured review covering historical context, technical claims, mathematical formulation, implementation, references, and internal consistency.

### Verified

Relevant technical claims, equations, implementation, and reproducibility requirements have been checked against primary or authoritative sources, with required tests passing where applicable.

### Publication-ready

The material has passed archival, historical, mathematical, implementation, reproducibility, citation, and editorial review and is ready for integration into the final book.

`File exists ≠ Research complete`

The metadata booleans mean:

* `archived = true` means material exists.
* `audited = true` means systematically reviewed.
* `verified = true` means relevant claims/technical material have been checked.
* `publication_ready = true` means final research and editorial review has passed.

Project-level counts must distinguish:

1. Papers Archived
2. Papers Audited
3. Papers Verified
4. Papers Publication-ready

Current project-level count status:

| Count | Current status |
| :--- | :--- |
| Papers Archived | 31 |
| Papers Audited | Not yet complete; no project-level audited count is claimed |
| Papers Verified | Not yet complete; no project-level verified count is claimed |
| Papers Publication-ready | Not yet complete; no project-level publication-ready count is claimed |

---

## Current Project Baseline

The current repository contains 31 archived papers represented in the master timeline.

* 31 papers are archived.
* 31 papers are represented in the master timeline.
* Historical accuracy review is not yet complete.
* Mathematical completeness review is not yet complete.
* Implementation verification is not yet complete.
* Diagram coverage is not yet complete.
* The book manuscript is not yet publication-ready.

The project is therefore:

`archivally complete but research-wise incomplete`

The 31 archived papers are not claimed to be verified or publication-ready.

---

## Controlled Project Roadmap

The project follows a fixed audit and integration sequence. New work is not added to the active scope outside this sequence.

### Phase 0 — Project Baseline

Establish canonical project status definitions, scope, and repository-wide consistency.

### Phase 1 — Repository Audit

Inspect the repository structure and determine what is complete, incomplete, inconsistent, redundant, or missing.

### Phase 2 — Historical Timeline Audit

Audit the master timeline chronologically and verify that each transition is historically and causally justified.

### Phase 3 — Paper-by-Paper Research Audit

Audit every archived paper in chronological order.

### Phase 4 — Mathematical Audit

Verify equations, assumptions, derivations, notation, and mathematical connections.

### Phase 5 — Implementation Audit

Review implementations, classify their depth, and verify selected implementations through reproducible tests.

### Phase 6 — Historical and Dependency Integration

Verify causal relationships between papers, methods, architectures, and research eras.

### Phase 7 — Diagrams, Comparisons, and Glossary

Complete the supporting research infrastructure required to make the archive understandable and navigable.

### Phase 8 — Book Integration

Convert the verified research archive into the structured book manuscript.

### Phase 9 — Final Integrity Review

Perform repository-wide consistency, citation, mathematical, implementation, historical, and editorial checks.

---

## Scope Control Rule

A new paper, technology, model, or research direction should not be added to the active scope merely because it is interesting or recent.

A new item should be added only if it:

1. closes a documented historical gap;
2. explains an important causal transition;
3. provides necessary mathematical or technical context;
4. is required to understand a later architectural development; or
5. materially improves reproducibility or verification.

Otherwise it remains outside the active scope until the relevant audit stage is reached.

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs.git
cd From-Biological-Neurons-to-LLMs
```

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Run Validation

```bash
python scripts/validate_repository.py
```

### Build Index & Knowledge Graph

```bash
python scripts/build_index.py
python scripts/build_graph.py
```

---

## Folder Structure

```text
From-Biological-Neurons-to-LLMs/
├── book/                    # LaTeX manuscript
├── research/                # The Archive
│   ├── papers/              # Per-paper folders (1943–2023)
│   ├── history/             # Narratives + dependency map
│   ├── graph/               # Knowledge graph (JSON/DOT/SVG)
│   ├── index.yaml           # Machine-readable index
│   ├── chronology/          # Master chronology
│   ├── comparisons/         # Paper comparisons
│   └── glossary/            # Terminology definitions
├── code/                    # NumPy implementations
├── bibliography/            # BibTeX sources
├── scripts/                 # Automation scripts
├── requirements-dev.txt     # Development dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks
└── README.md
```

---

## Current Papers Archived (31)

| # | Year | Paper | Project State |
| :--- | :--- | :--- | :--- |
| 1 | 1943 | McCulloch & Pitts | Archived |
| 2 | 1949 | Hebb | Archived |
| 3 | 1950 | Turing | Archived |
| 4 | 1958 | Rosenblatt | Archived |
| 5 | 1960 | Widrow & Hoff | Archived |
| 6 | 1969 | Minsky & Papert | Archived |
| 7 | 1974 | Werbos | Archived |
| 8 | 1980 | Fukushima | Archived |
| 9 | 1982 | Hopfield | Archived |
| 10 | 1986 | Rumelhart, Hinton & Williams | Archived |
| 11 | 1989 | LeCun CNN | Archived |
| 12 | 1990 | Jordan Network | Archived |
| 13 | 1991 | Elman Network | Archived |
| 14 | 1997 | LSTM | Archived |
| 15 | 1998 | LeNet-5 | Archived |
| 16 | 2012 | AlexNet | Archived |
| 17 | 2014 | Seq2Seq | Archived |
| 18 | 2014 | GAN | Archived |
| 19 | 2015 | ResNet | Archived |
| 20 | 2017 | Transformer | Archived |
| 21 | 2018 | GPT | Archived |
| 22 | 2018 | BERT | Archived |
| 23 | 2019 | GPT-2 | Archived |
| 24 | 2020 | GPT-3 | Archived |
| 25 | 2022 | InstructGPT | Archived |
| 26 | 2022 | ChatGPT | Archived |
| 27 | 2023 | GPT-4 | Archived |
| 28 | 2023 | LLaMA | Archived |
| 29 | 2023 | Llama 2 | Archived |
| 30 | 2023 | Mistral 7B | Archived |
| 31 | 2024 | Mixtral 8x7B | Archived |

---

## Contributing

Please read `CONTRIBUTING.md` and `STYLE_GUIDE.md` before submitting corrections.

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. See `LICENSE` for details.

---

**Last Updated:** 2026-07-18