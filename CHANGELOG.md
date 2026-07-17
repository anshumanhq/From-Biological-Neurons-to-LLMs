# Changelog

All notable changes to this project (manuscript, code, and research infrastructure) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the codebase and infrastructure milestones. 
The book manuscript itself will follow a separate `v1.0` release when fully drafted.

---

## [Unreleased]

### Planned (Phase 2+)
- Complete LaTeX project setup (`book/main.tex`, `latexmkrc`).
- Draft Chapter 1: The Brain Before AI.
- Draft Chapter 2: McCulloch-Pitts Neuron.
- Generate TikZ figures for all milestone architectures.
- Populate all 100+ paper archives with `notes.md`.

---

## [0.1.0] – 2026-07-17

### Added : Infrastructure & Research Backbone (Phase 0 & Phase 1)

#### Project Scaffolding
- Created master folder structure: `/book`, `/research`, `/code`, `/bibliography`, `/figures`, `/docs`, `/scripts`.
- Root files established: `README.md`, `STYLE_GUIDE.md`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`.

#### Literature Collection (Primary Sources)
- Collected and verified 100+ academic sources spanning 1943–2026.
- Core milestone papers archived with verified DOIs:
  - McCulloch & Pitts (1943)
  - Hebb (1949)
  - Rosenblatt (1958)
  - Minsky & Papert (1969)
  - Rumelhart, Hinton & Williams (1986)
  - Vaswani et al. (2017)
  - Brown et al. (GPT-3, 2020)

#### Research Archive Structure (Paper-per-Folder)
- Defined master template for `research/papers/{YYYY_Author_ShortTitle}/`.
- Scaffolded folders for the 4 milestone papers:
  - `1943_McCulloch_Pitts/`
  - `1949_Hebb_Organization/`
  - `1958_Rosenblatt_Perceptron/`
  - `2017_Vaswani_Attention/`

#### Historical Timeline
- Completed `ai_timeline_1943_2026.md` spanning 8 distinct eras:
  - Foundations (1943–1956)
  - Golden Age (1956–1973)
  - First AI Winter (1973–1980)
  - Connectionist Revival (1980–1997)
  - Deep Learning Revolution (2006–2016)
  - Transformer & LLM Era (2017–2021)
  - Generative Mainstream (2022–2023)
  - Agentic Shift (2024–2026)

#### Author Biographies
- Completed biographical entries for:
  - Warren McCulloch, Walter Pitts, Donald Hebb, Frank Rosenblatt
  - John Hopfield, Geoffrey Hinton, Yann LeCun, Yoshua Bengio
  - Jürgen Schmidhuber, Ashish Vaswani, Ilya Sutskever, Sam Altman

#### Equation Extraction
- Identified and documented all canonical equations from primary sources.
- LaTeX bank created for: MP Threshold, Hebbian update, Perceptron rule, Hopfield Energy, Backprop Delta Rule, LSTM gates, Scaled Dot-Product Attention, Positional Encoding.

#### Citation Verification
- Cross-checked all DOIs against CrossRef.
- Detected and documented 4 major historical misattributions:
  - Werbos (1974) correctly identified as backpropagation originator (not Rumelhart et al.).
  - Minsky & Papert (1969) limited to *single-layer* perceptrons only.
  - Hebb (1949) correctly classified as unsupervised/associative learning.
  - Turing (1950) clarified as proposing RL/child machine, not neural networks.

#### NumPy Scratch Implementations
- Delivered 7 standalone, dependency-free implementations:
  - `mcculloch_pitts()` – threshold logic
  - `hebbian_update()` – associative weight modification
  - `PerceptronScratch` – Rosenblatt's classifier
  - `hopfield_energy()` + `hopfield_update()` – associative memory
  - `MLP_Scratch` – 2-layer network with manual backpropagation
  - `lstm_cell()` – gated recurrent unit
  - `scaled_dot_product_attention()` – core Transformer operation

#### Bibliography
- Created `bibliography/` folder.
- Initial `papers.bib` populated with 7 fully verified entries, each containing DOI/ISBN fields.

#### Style & Governance
- Finalized `STYLE_GUIDE.md` with strict rules for:
  - Heading hierarchy
  - Equation numbering (`(Chapter.Equation)`)
  - Figure/Table labeling
  - Citation format (Author-Year)
  - Paper analysis templates
- Finalized `CONTRIBUTING.md` with academic PR guidelines and Issue labels.

---

### Changed
- (None – this is the initial release)

---

### Deprecated
- (None)

---

### Removed
- (None)

---

### Security
- (None – no deployment dependencies)

---

## [0.0.0] – 2026-07-16 (Pre-Infrastructure)
- Initial empty repository created.
- Project vision conceptualized.

---

*The next milestone (v0.2.0) will focus on the LaTeX project setup and drafting the first two chapters.*