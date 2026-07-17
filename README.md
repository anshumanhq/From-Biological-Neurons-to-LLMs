# From Biological Neurons to Large Language Models

**Version:** 0.3.0 (Infrastructure Complete)  
**License:** CC BY-NC-ND 4.0

---

## Status Badges

[![Validation](https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs/actions/workflows/validate.yml/badge.svg)](https://github.com/anshumanhq/From-Biological-Neurons-to-LLMs/actions/workflows/validate.yml)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Papers Archived](https://img.shields.io/badge/Papers%20Archived-11-brightgreen)](research/index.yaml)
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
| **v0.3.0** | **Infrastructure Automation** | ✅ **Complete** |
| **v0.4.0** | Classical Neural Networks (1943–1998) | 🚧 In Progress |
| **v0.5.0** | Deep Learning Era (2006–2015) | 📅 Planned |
| **v1.0.0** | Transformer → GPT → LLMs Complete | 📅 Planned |

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/yourusername/From-Biological-Neurons-to-LLMs.git
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
│   ├── papers/              # Per-paper folders (1943–1989)
│   ├── history/             # Narratives + dependency map
│   ├── graph/               # Knowledge graph (JSON/DOT/SVG)
│   ├── index.yaml           # Machine-readable index
│   └── PAPER_TEMPLATE.md    # 18-section template
├── code/                    # NumPy implementations
├── bibliography/            # BibTeX sources
├── scripts/                 # Automation scripts
├── requirements-dev.txt     # Development dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks
└── README.md
```

---

## Current Papers Archived (11)

| Year | Paper | Status |
| :--- | :--- | :--- |
| 1943 | McCulloch & Pitts | ✅ Complete |
| 1949 | Hebb | ✅ Complete |
| 1950 | Turing | ✅ Complete |
| 1958 | Rosenblatt | ✅ Complete |
| 1960 | Widrow & Hoff | ✅ Complete |
| 1969 | Minsky & Papert | ✅ Complete |
| 1974 | Werbos | ✅ Complete |
| 1980 | Fukushima | ✅ Complete |
| 1982 | Hopfield | ✅ Complete |
| 1986 | Rumelhart, Hinton & Williams | ✅ Complete |
| 1989 | LeCun | ✅ Complete |

---

## Next Milestone

**v0.4.0 — Classical Neural Networks (1943–1998)**

- 1990: Jordan Network
- 1991: Elman Network
- 1997: LSTM
- 1998: LeNet-5

---

## Contributing

Please read `CONTRIBUTING.md` and `STYLE_GUIDE.md` before submitting corrections.

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. See `LICENSE` for details.

---

**Last Updated:** 2026-07-17