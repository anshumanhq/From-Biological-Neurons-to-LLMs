# Changelog

All notable changes to this project (manuscript, code, and research infrastructure) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the codebase and infrastructure milestones. 
The book manuscript itself will follow a separate `v1.0` release when fully drafted.

---

## [Unreleased]

### Planned (v0.7.0+)
- Populate BERT (2018) archive
- Populate Mixtral 8x7B (2024) archive
- Populate Word2Vec (2013) archive
- Populate Bahdanau Attention (2014) archive
- Populate Batch Normalization (2015) archive
- Populate RAG / Tool Use / Agentic AI (2024–2026) overview
- Draft Chapter 1: The Brain Before AI
- Draft Chapter 2: McCulloch-Pitts Neuron
- Generate TikZ figures for all milestone architectures
- Complete LaTeX project setup (`book/main.tex`, `latexmkrc`)

---

## [0.6.0] – 2026-07-18

### Added : Modern LLMs & Open-Weight Era (2023)

#### New Paper Archives
- **2023_OpenAI_GPT4** — Capability leap, reasoning, multimodal potential
- **2023_Meta_LLaMA** — Open-weight research access, efficient foundation models
- **2023_Meta_Llama2** — Open-weight commercial release, RLHF chat models
- **2023_Mistral_7B** — Efficiency innovations (GQA, SWA)

#### Historical Narratives
- `narrative_1989_1997.md` — The Architecture Era
- `narrative_1998_2012.md` — Deep Learning Maturation
- `narrative_2013_2017.md` — Modern Neural Networks
- `narrative_2018_2026.md` — The LLM Era

#### Infrastructure Improvements
- **Knowledge Graph** — Updated to 29 nodes, 110 edges
- **Master Timeline** — Updated with all 29 papers
- **README** — Updated paper count and roadmap
- **CHANGELOG** — Updated with all recent additions

---

## [0.5.0] – 2026-07-18

### Added : Deep Learning Era & LLM Era (2012–2022)

#### New Paper Archives
- **2012_Krizhevsky_AlexNet** — ImageNet breakthrough with ReLU, dropout, GPU training
- **2014_Sutskever_Seq2Seq** — Encoder-decoder architecture for sequence transduction
- **2014_Goodfellow_GAN** — Adversarial training for generative models
- **2015_He_ResNet** — Residual learning for very deep networks
- **2017_Vaswani_Transformer** — Attention-based sequence transduction
- **2018_Radford_GPT** — Generative pre-training and transfer learning
- **2019_Radford_GPT2** — Zero-shot transfer at scale (1.5B parameters)
- **2020_Brown_GPT3** — Few-shot learning at 175B scale
- **2022_Ouyang_InstructGPT** — RLHF for alignment and instruction following
- **2022_OpenAI_ChatGPT** — Conversational AI product milestone

#### Infrastructure Improvements
- **Knowledge Graph** — Updated to 25 nodes, 91 edges
- **Master Timeline** — Updated with all 25 papers
- **Dependency Map** — Extended to cover LLM era
- **README** — Updated paper count and roadmap

---

## [0.4.0] – 2026-07-18

### Added : Classical Neural Networks (1943–1998)

#### New Paper Archives
- **1990_Jordan_Network** — First practical recurrent network with context units
- **1991_Elman_Network** — Simple RNN with hidden-state recurrence
- **1997_Hochreiter_LSTM** — Original LSTM with CEC, input/output gates (no forget gate)
- **1998_LeCun_LeNet5** — Refined CNN architecture (7 layers)

#### Infrastructure Improvements
- **Glossary** — Added 10 entries
- **Comparisons** — Added `jordan_vs_elman.md` and `lstm_1997_vs_forget_gate.md`
- **Chronology** — Added `neural_network_history.md` master timeline
- **Metadata Schema** — Added `schema_version: "1.0"` to all metadata files
- **Reproducibility** — Added reproducibility fields
- **Implementation Status** — Added structured status fields
- **Paper Source** — All papers now have `paper_source.md`

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
- **6 Robust Tests** for NumPy implementations
- **Dynamic module loading** — Tests use `importlib` to load implementations

#### ADALINE Implementation Fix
- Switched from 0/1 targets to **bipolar targets (-1/1)** in `implementation_historical.py`

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
- 29 milestone papers archived (1943–2023)
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

*The next milestone (v0.7.0) will focus on completing the remaining Transformer-era landmarks:*
- *2018: BERT (bidirectional pre-training)*
- *2024: Mixtral 8x7B (Mixture of Experts)*
- *2013: Word2Vec*
- *2014: Bahdanau Attention*
- *2015: Batch Normalization*
- *2024–2026: RAG / Tool Use / Agentic AI overview*