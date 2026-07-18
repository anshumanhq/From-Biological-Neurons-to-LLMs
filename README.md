# From Biological Neurons to Large Language Models

**Version:** 0.6.0 (Modern LLMs & Open-Weight Era)  
**License:** CC BY-NC-ND 4.0

---

## Status Badges

[![Validation](https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs/actions/workflows/validate.yml/badge.svg)](https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs/actions/workflows/validate.yml)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Papers Archived](https://img.shields.io/badge/Papers%20Archived-29-brightgreen)](research/index.yaml)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)

---

## Book Objective

This project is a comprehensive, historically accurate, and technically rigorous chronicle of the evolution of Artificial Intelligence. It traces the mathematical and biological lineage from the McCulloch-Pitts neuron (1943) to modern Transformer-based Large Language Models (2026).

Unlike typical history books, this repository treats the book as a **living research archive**. Every claim is tied to a primary source, every equation is re-implemented from scratch in NumPy, and every chapter is version-controlled.

---

## Repository Roadmap

| Phase | Period | Status |
| :--- | :--- | :--- |
| **v0.1.0** | Repository Governance | ✅ Complete |
| **v0.2.0** | LaTeX Infrastructure | ✅ Complete |
| **v0.2.1** | Repository Infrastructure | ✅ Complete |
| **v0.3.0** | Infrastructure Automation | ✅ Complete |
| **v0.4.0** | Classical Neural Networks (1943–1998) | ✅ Complete |
| **v0.5.0** | Deep Learning & LLM Era (2012–2022) | ✅ Complete |
| **v0.6.0** | Modern LLMs & Open-Weight Era (2023) | ✅ **Complete** |
| **v0.7.0** | Remaining Transformer Landmarks (BERT, Mixtral, etc.) | 🚧 In Progress |
| **v1.0.0** | Book Manuscript Complete | 📅 Planned |

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

## Current Papers Archived (29)

| # | Year | Paper | Status |
| :--- | :--- | :--- | :--- |
| 1 | 1943 | McCulloch & Pitts | ✅ Complete |
| 2 | 1949 | Hebb | ✅ Complete |
| 3 | 1950 | Turing | ✅ Complete |
| 4 | 1958 | Rosenblatt | ✅ Complete |
| 5 | 1960 | Widrow & Hoff | ✅ Complete |
| 6 | 1969 | Minsky & Papert | ✅ Complete |
| 7 | 1974 | Werbos | ✅ Complete |
| 8 | 1980 | Fukushima | ✅ Complete |
| 9 | 1982 | Hopfield | ✅ Complete |
| 10 | 1986 | Rumelhart, Hinton & Williams | ✅ Complete |
| 11 | 1989 | LeCun CNN | ✅ Complete |
| 12 | 1990 | Jordan Network | ✅ Complete |
| 13 | 1991 | Elman Network | ✅ Complete |
| 14 | 1997 | LSTM | ✅ Complete |
| 15 | 1998 | LeNet‑5 | ✅ Complete |
| 16 | 2012 | AlexNet | ✅ Complete |
| 17 | 2014 | Seq2Seq | ✅ Complete |
| 18 | 2014 | GAN | ✅ Complete |
| 19 | 2015 | ResNet | ✅ Complete |
| 20 | 2017 | Transformer | ✅ Complete |
| 21 | 2018 | GPT | ✅ Complete |
| 22 | 2019 | GPT‑2 | ✅ Complete |
| 23 | 2020 | GPT‑3 | ✅ Complete |
| 24 | 2022 | InstructGPT | ✅ Complete |
| 25 | 2022 | ChatGPT | ✅ Complete |
| 26 | 2023 | GPT‑4 | ✅ Complete |
| 27 | 2023 | LLaMA | ✅ Complete |
| 28 | 2023 | Llama 2 | ✅ Complete |
| 29 | 2023 | Mistral 7B | ✅ Complete |

---

## Next Milestone

**v0.7.0 — Remaining Transformer Landmarks**

- 2018: BERT
- 2024: Mixtral 8x7B
- 2013: Word2Vec
- 2014: Bahdanau Attention
- 2015: Batch Normalization
- 2024–2026: RAG / Tool Use / Agentic AI

---

## Contributing

Please read `CONTRIBUTING.md` and `STYLE_GUIDE.md` before submitting corrections.

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. See `LICENSE` for details.

---

**Last Updated:** 2026-07-18