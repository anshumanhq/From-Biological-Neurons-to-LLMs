# Changelog

All notable changes to this project (manuscript, code, and research infrastructure) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the codebase and infrastructure milestones. 
The book manuscript itself will follow a separate `v1.0` release when fully drafted.

---

## [Unreleased]

### Planned (v0.4.0+)
- Populate Elman Network (1991) archive
- Populate LSTM (1997) archive
- Populate AlexNet (2012) archive
- Populate Seq2Seq (2014) archive
- Populate GAN (2014) archive
- Populate ResNet (2015) archive
- Populate Transformer (2017) archive
- Draft Chapter 1: The Brain Before AI
- Draft Chapter 2: McCulloch-Pitts Neuron
- Generate TikZ figures for all milestone architectures
- Complete LaTeX project setup (`book/main.tex`, `latexmkrc`)

---

## [0.3.0] – 2026-07-17

### Added : Infrastructure Automation & Paper Archives

#### New Paper Archives
- **1990_Jordan_Network** — First practical recurrent network with context units
- **1998_LeCun_LeNet5** — Refined CNN architecture (7 layers: conv → pool → conv → pool → FC → FC → output)

#### Infrastructure Improvements
- **Makefile** — Standardised commands: `make validate`, `make test`, `make index`, `make graph`, `make all`
- **Pre-commit hooks** — Added `.pre-commit-config.yaml` with:
  - trailing-whitespace removal
  - end-of-file-fixer
  - YAML validation
  - Ruff linting (with `--fix`)
  - Black formatting
- **Validation script** — `scripts/validate_repository.py` now checks:
  - Folder name pattern (`YYYY_Author_ShortTitle`)
  - Required files per paper
  - Empty files detection
  - Metadata fields completeness
  - DOI format validation
  - Implementation filename consistency
- **Index generator** — `scripts/build_index.py` generates `research/index.yaml` automatically
- **Graph generator** — `scripts/build_graph.py` produces `knowledge_graph.json`, `.dot`, and `.svg`
- **GitHub Actions** — CI workflow runs validation, index build, graph build, and pytest tests

#### Testing
- **6 Robust Tests** for NumPy implementations:
  - Hebbian update (weights finite, shape preserved)
  - Perceptron XOR (linear separability → fails gracefully)
  - Perceptron AND (converges correctly)
  - ADALINE AND (bipolar targets, loss decreases, correct boundary)
  - Hopfield recovery (pattern recall from corrupted input)
  - XOR backprop (loss decreases, ≥75% accuracy)
- **Dynamic module loading** — Tests use `importlib` to load implementations from file paths
- **Random seeds** — All tests are deterministic (`np.random.seed(42)`)

#### ADALINE Implementation Fix
- Switched from 0/1 targets to **bipolar targets (-1/1)** in `implementation_historical.py`
- Matches the original Widrow-Hoff formulation
- Now correctly classifies AND gate with 4/4 accuracy

#### Documentation
- `paper_source.md` replaces empty `original.pdf` placeholders (with DOI, publisher, access notes)
- All paper folders now contain:
  - `metadata.yaml` (machine-readable)
  - `bibliography.bib` (BibTeX entry)
  - `paper_source.md` (access info)
  - `README.md` (navigation)
  - `implementation_historical.py` + `implementation_modern.py` (where applicable)

---

## [0.2.0] – 2026-07-17

### Added : LaTeX Infrastructure

#### Book Skeleton
- `book/main.tex` — Master entry point
- `book/preamble.tex` — All LaTeX packages and settings
- `book/references.bib` — Merged bibliography
- `book/frontmatter/` — Title, abstract, dedication, preface
- `book/chapters/` — Empty chapter templates
- `book/latexmkrc` — Auto-compilation settings
- `book/Makefile` — Build automation

#### Bibliography
- `bibliography/papers.bib`, `bibliography/books.bib`, `bibliography/web.bib`
- 11 verified entries with DOIs/ISBNs

---

## [0.1.0] – 2026-07-17

### Added : Infrastructure & Research Backbone (Phase 0 & Phase 1)

#### Project Scaffolding
- Master folder structure: `/book`, `/research`, `/code`, `/bibliography`, `/figures`, `/docs`, `/scripts`
- Root files: `README.md`, `STYLE_GUIDE.md`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`

#### Literature Collection (Primary Sources)
- Collected and verified 100+ academic sources spanning 1943–2026
- Core milestone papers archived with verified DOIs:
  - McCulloch & Pitts (1943)
  - Hebb (1949)
  - Rosenblatt (1958)
  - Widrow & Hoff (1960)
  - Minsky & Papert (1969)
  - Werbos (1974)
  - Fukushima (1980)
  - Hopfield (1982)
  - Rumelhart, Hinton & Williams (1986)
  - LeCun (1989)

#### Research Archive Structure (Paper-per-Folder)
- Master template: `research/papers/{YYYY}_{AuthorLastName}_{ShortTitle}/`
- 11 completed paper archives (1943–1989)

#### Historical Narratives
- `narrative_1943_1958.md` — Foundations
- `narrative_1960_1969.md` — The Mathematical Turn
- `narrative_1974_1986.md` — The Revival
- `dependency_map.md` — Intellectual lineage of all papers

#### Author Biographies
- McCulloch, Pitts, Hebb, Rosenblatt, Hopfield, Hinton, LeCun, Bengio, Schmidhuber, Vaswani, Sutskever, Altman

#### Equation Extraction
- All canonical equations from primary sources in LaTeX format

#### Citation Verification
- Cross-checked all DOIs against CrossRef
- Documented 4 major historical misattributions (Werbos, Minsky & Papert, Hebb, Turing)

#### NumPy Scratch Implementations
- 7 standalone implementations: MP neuron, Hebbian update, Perceptron, Hopfield energy, MLP with backprop, LSTM cell, Scaled Dot-Product Attention

#### Style & Governance
- `STYLE_GUIDE.md` — Heading hierarchy, equation numbering, citation format, paper analysis templates
- `CONTRIBUTING.md` — Academic PR guidelines and Issue labels

---

## [0.0.0] – 2026-07-16 (Pre-Infrastructure)
- Initial empty repository created
- Project vision conceptualized

---

*The next milestone (v0.4.0) will focus on completing the Classical Neural Networks era:*
- *1991: Elman Network*
- *1997: LSTM*
- *2012: AlexNet*
- *2014: Seq2Seq, GAN*
- *2015: ResNet*
- *2017: Transformer*