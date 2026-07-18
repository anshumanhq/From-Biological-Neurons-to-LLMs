# Changelog

All notable changes to this project (manuscript, code, and research infrastructure) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the codebase and infrastructure milestones. 
The book manuscript itself will follow a separate `v1.0` release when fully drafted.

---

## [Unreleased]

### Planned (v0.5.0+)
- Populate GAN (2014) archive
- Populate ResNet (2015) archive
- Populate Transformer (2017) archive
- Populate GPT (2018) archive
- Populate BERT (2018) archive
- Draft Chapter 1: The Brain Before AI
- Draft Chapter 2: McCulloch-Pitts Neuron
- Generate TikZ figures for all milestone architectures

---

## [0.4.0] – 2026-07-18

### Added : Classical Neural Networks (1943–1998) & Deep Learning Era (2012–2014)

#### New Paper Archives
- **1990_Jordan_Network** — First practical recurrent network with context units
- **1991_Elman_Network** — Simple RNN with hidden-state recurrence
- **1997_Hochreiter_LSTM** — Original LSTM with CEC, input/output gates (no forget gate)
- **1998_LeCun_LeNet5** — Refined CNN architecture (7 layers)
- **2012_Krizhevsky_AlexNet** — ImageNet breakthrough with ReLU, dropout, GPU training
- **2014_Sutskever_Seq2Seq** — Encoder-decoder architecture for sequence transduction

#### Infrastructure Improvements
- **Glossary** — Added 10 entries (activation_function, backpropagation, bptt, context_unit, hidden_state, teacher_forcing, vanishing_gradient, exploding_gradient, attention, embedding)
- **Comparisons** — Added `jordan_vs_elman.md` and `lstm_1997_vs_forget_gate.md`
- **Chronology** — Added `neural_network_history.md` master timeline
- **Metadata Schema** — Added `schema_version: "1.0"` to all metadata files
- **Reproducibility** — Added reproducibility fields (python, numpy, last_tested)
- **Implementation Status** — Added structured status fields (historical: status/verified, modern: status/verified)
- **Paper Source** — All papers now have `paper_source.md` (replacing empty PDF placeholders)

---

## [0.3.0] – 2026-07-17

### Added : Infrastructure Automation & Paper Archives

#### New Paper Archives
- **1990_Jordan_Network** — First practical recurrent network with context units
- **1998_LeCun_LeNet5** — Refined CNN architecture

#### Infrastructure Improvements
- **Makefile** — Standardised commands: `make validate`, `make test`, `make index`, `make graph`, `make all`
- **Pre-commit hooks** — Added `.pre-commit-config.yaml`
- **Validation script** — `scripts/validate_repository.py` with comprehensive checks
- **Index generator** — `scripts/build_index.py` generates `research/index.yaml`
- **Graph generator** — `scripts/build_graph.py` produces knowledge graph
- **GitHub Actions** — CI workflow with validation, index, graph, and tests

#### Testing
- **6 Robust Tests** for NumPy implementations (Hebbian, Perceptron, ADALINE, Hopfield, Backprop)
- **Dynamic module loading** — Tests use `importlib` to load implementations

#### ADALINE Implementation Fix
- Switched from 0/1 targets to **bipolar targets (-1/1)** in `implementation_historical.py`
- Matches the original Widrow-Hoff formulation

#### Documentation
- `paper_source.md` replaces empty `original.pdf` placeholders
- All paper folders now contain metadata, bibliography, paper_source, README, and implementations

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

---

## [0.1.0] – 2026-07-17

### Added : Infrastructure & Research Backbone

#### Project Scaffolding
- Master folder structure: `/book`, `/research`, `/code`, `/bibliography`, `/figures`, `/docs`, `/scripts`
- Root files: `README.md`, `STYLE_GUIDE.md`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`

#### Literature Collection
- 17 milestone papers archived (1943–2014)
- Verified DOIs and metadata

#### Historical Narratives
- `narrative_1943_1958.md` — Foundations
- `narrative_1960_1969.md` — The Mathematical Turn
- `narrative_1974_1986.md` — The Revival
- `dependency_map.md` — Intellectual lineage

#### Author Biographies
- McCulloch, Pitts, Hebb, Rosenblatt, Hopfield, Hinton, LeCun, Bengio, Schmidhuber, Vaswani, Sutskever, Altman

#### NumPy Scratch Implementations
- 7 standalone implementations: MP neuron, Hebbian update, Perceptron, Hopfield energy, MLP with backprop, LSTM cell, Scaled Dot-Product Attention

---

## [0.0.0] – 2026-07-16 (Pre-Infrastructure)
- Initial empty repository created
- Project vision conceptualized

---

*The next milestone (v0.5.0) will focus on completing the Deep Learning Era:*
- *2014: GAN*
- *2015: ResNet*
- *2017: Transformer*
- *2018: GPT, BERT*
- *2020: GPT-3*
- *2022: InstructGPT / RLHF*