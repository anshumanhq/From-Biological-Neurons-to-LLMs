# Dependency Map: From Biological Neurons to LLMs

**Project:** From Biological Neurons to Large Language Models  
**Last Updated:** 2026-07-17  
**Purpose:** Summarises every paper's intellectual lineage — what it built on, what it solved, what it introduced, and what it enabled.

---

## Phase 1: Foundations (1943–1958)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **1943 – McCulloch & Pitts** | None (foundational) | Formal logic computation using neurons | Threshold logic unit (binary neuron) | All subsequent neural network research |
| **1949 – Hebb** | 1943 – McCulloch & Pitts | Synaptic plasticity mechanism | Hebbian learning rule ("cells that fire together, wire together") | Associative memory, unsupervised learning |
| **1950 – Turing** | 1936 – Turing Machine | Philosophical framing of machine intelligence | Imitation Game (Turing Test), Child Machine concept | Modern AI evaluation, reinforcement learning |
| **1958 – Rosenblatt** | 1943, 1949 | Trainable linear classifier | Perceptron (first trainable neural network) | Supervised learning in neural networks |

---

## Phase 2: The Mathematical Turn (1960–1969)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **1960 – Widrow & Hoff** | 1958 – Rosenblatt | Continuous error minimization | LMS (Delta Rule) / Stochastic Gradient Descent | Gradient-based optimization in neural nets |
| **1969 – Minsky & Papert** | 1958, 1960 | Single-layer limitations (XOR, parity) | Mathematical rigor in neural network analysis | Multi-layer network challenge (backpropagation) |

---

## Phase 3: The Revival (1974–1986)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **1974 – Werbos** | 1960 – Widrow & Hoff | Credit assignment for hidden layers | Backpropagation (reverse-mode differentiation) | Multi-layer network training |
| **1980 – Fukushima** | 1962 – Hubel & Wiesel | Shift invariance in pattern recognition | Neocognitron (local receptive fields, pooling) | Convolutional neural networks |
| **1982 – Hopfield** | 1949 – Hebb | Associative memory with stable dynamics | Energy-based recurrent networks | Boltzmann Machines, modern Hopfield networks |
| **1986 – Rumelhart, Hinton & Williams** | 1974 – Werbos, 1982 – Hopfield | Practical multi-layer training | Popularised backpropagation | Deep learning revolution |

---

## Phase 4: The Architecture Era (1989–1997)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **1989 – LeCun** | 1980 – Fukushima, 1986 – Rumelhart | Handwritten digit recognition | Supervised CNN (backprop + convolution) | LeNet-5, AlexNet, modern computer vision |
| **1990 – Jordan** | 1982 – Hopfield | Recurrent networks for sequential data | Jordan Network (recurrent connections) | Sequence modelling |
| **1991 – Elman** | 1990 – Jordan | Simple recurrent networks | Elman Network (context units) | RNNs for language |
| **1997 – LSTM** | 1991 – Elman | Vanishing gradients in RNNs | LSTM (forget gates, memory cells) | Seq2Seq, Transformers, modern NLP |

---

## Phase 5: Deep Learning Maturation (1998–2012)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **1998 – LeNet-5** | 1989 – LeCun | Large-scale digit recognition | Mature CNN architecture | Modern deep learning |
| **2006 – Deep Belief Nets** | 1986 – Backprop, 1982 – Hopfield | Training deep generative models | Greedy layer-wise pre-training | Deep learning revival (2006) |
| **2012 – AlexNet** | 1998 – LeNet-5 | ImageNet classification at scale | GPU-accelerated deep CNNs, ReLU, Dropout | Modern computer vision, AI boom |

---

## Phase 6: Modern Neural Networks (2013–2016)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **2014 – Seq2Seq** | 1997 – LSTM | Neural machine translation | Encoder-decoder architecture | Modern NLP, translation, summarization |
| **2014 – GAN** | 2006 – DBN, 1982 – Hopfield | Generative modelling | Generative Adversarial Networks | Image generation, creative AI |
| **2015 – ResNet** | 2012 – AlexNet | Training very deep networks | Residual connections (skip connections) | Extremely deep networks (100+ layers) |

---

## Phase 7: The Transformer & LLM Era (2017–2026)

| Paper | Built On | Solved | Introduced | Enabled |
| :--- | :--- | :--- | :--- | :--- |
| **2017 – Transformer** | 2014 – Seq2Seq (Attention), 1997 – LSTM | Sequential bottleneck in RNNs | Self-attention mechanism, Positional encoding | GPT, BERT, all modern LLMs |
| **2018 – GPT** | 2017 – Transformer | Generative pre-training | Causal language modelling | Modern LLMs |
| **2018 – BERT** | 2017 – Transformer | Bidirectional pre-training | Masked language modelling | Deep bidirectional representations |
| **2020 – GPT-3** | 2018 – GPT | Few-shot learning at scale | Scaling laws, Emergent abilities | Foundation models |
| **2022 – InstructGPT / RLHF** | 2020 – GPT-3 | Aligning models with human intent | Reinforcement Learning from Human Feedback | ChatGPT, Claude, Gemini |
| **2026 – Agentic AI** | All previous | Coordinated execution over time | Tools + Memory + Governance | Modern AI systems (agents) |

---

## Visual Summary

```text
1943 ──► 1949 ──► 1958 ──► 1960 ──► 1969
           │         │         │         │
           │         │         │         ▼
           │         │         │     1974 ──► 1986
           │         │         │         │         │
           │         │         │         │         ▼
           │         │         │         │     1989 ──► 1998 ──► 2012
           │         │         │         │         │         │         │
           │         │         │         │         │         │         ▼
           │         │         │         │         │         │     2017 ──► 2018 ──► 2020 ──► 2022
           │         │         │         │         │         │         │         │         │         │
           │         │         │         │         │         │         │         │         │         ▼
           │         │         │         │         │         │         │         │         │     2024–2026
           │         │         │         │         │         │         │         │         │     (Agentic AI)
           │         │         │         │         │         │         │         │         │
           │         │         │         │         │         │         │         │         ▼
           │         │         │         │         │         │         │         │    2019 – GPT-2
           │         │         │         │         │         │         │         │
           │         │         │         │         │         │         │         ▼
           │         │         │         │         │         │         │    2020 – GPT-3
           │         │         │         │         │         │         │
           │         │         │         │         │         │         ▼
           │         │         │         │         │         │    2022 – InstructGPT/RLHF
           │         │         │         │         │         │
           │         │         │         │         │         ▼
           │         │         │         │         │    2024–2026 – Agentic AI
           │         │         │         │         │
           │         │         │         │         ▼
           │         │         │         │    2023 – GPT-4
           │         │         │         │
           │         │         │         ▼
           │         │         │     1997 – LSTM
           │         │         │
           │         │         ▼
           │         │     1991 – Elman Network
           │         │
           │         ▼
           │     1990 – Jordan Network
           │
           ▼
     1980 – Fukushima (Neocognitron)
           │
           ▼
     1982 – Hopfield Network
```

---

## Summary

| Phase | Period | Key Contribution | Status |
| :--- | :--- | :--- | :--- |
| Foundations | 1943–1958 | Biological inspiration & first learning machines | ✅ Complete |
| Mathematical Turn | 1960–1969 | Optimization theory & single-layer limits | ✅ Complete |
| The Revival | 1974–1986 | Backpropagation, CNNs, physics-based models | ✅ Complete |
| Architecture Era | 1989–1997 | CNNs, RNNs, LSTM | 🚧 Planned |
| Deep Learning Maturation | 1998–2012 | Deep learning revival, AlexNet | 📅 Planned |
| Modern Neural Networks | 2013–2016 | Seq2Seq, GANs, ResNet | 📅 Planned |
| Transformer & LLM Era | 2017–2026 | Attention, GPT, scaling, agents | 📅 Planned |
