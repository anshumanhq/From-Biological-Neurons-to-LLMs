# Historical Narrative: The LLM Era – From GPT to Agentic AI (2018–2026)

**Author:** Anshuman Singh  
**Project:** From Biological Neurons to Large Language Models  
**Archive Status:** Milestone 7 – In Progress (GPT 2018, GPT-2 2019, GPT-3 2020, InstructGPT 2022, ChatGPT 2022, GPT-4 2023, LLaMA 2023)  
**Last Updated:** 2026-07-18

---

## Introduction

The Transformer's arrival in 2017 was a catalyst. Within a year, it had been repurposed for language modelling, and within a few years, scaling it to billions of parameters revealed capabilities that few had anticipated. This narrative traces the evolution of large language models from GPT (2018) through GPT‑4 (2023) and the open‑weight ecosystem catalysed by LLaMA (2023). The period ends with the emergence of agentic AI, where language models interact with the world through tools, memory, and reasoning.

---

## 1. GPT and GPT‑2: The Birth of Generative Pre‑training (2018–2019)

Radford et al. introduced **GPT** (Generative Pre‑training) in 2018, showing that a Transformer decoder, pre‑trained on a large corpus (BooksCorpus) and then fine‑tuned, could achieve state‑of‑the‑art results on a variety of NLP tasks. The model was modest (117M parameters) but established the **pre‑train then fine‑tune** paradigm.

In 2019, **GPT‑2** (1.5B parameters) demonstrated that scaling improved zero‑shot performance: the model could perform tasks without fine‑tuning, simply by conditioning on a prompt. The release was staged due to concerns about misuse, but the zero‑shot results were remarkable.

**Key insight:** Language modelling, at scale, is a general‑purpose learning framework.

---

## 2. GPT‑3: Few‑Shot Learning at Scale (2020)

Brown et al. scaled GPT‑3 to 175B parameters and evaluated it in zero‑shot, one‑shot, and few‑shot settings. The model demonstrated that **in‑context learning**—providing demonstrations in the prompt—dramatically improved performance. Few‑shot learning closed the gap with fine‑tuned models on many tasks.

**Technical contributions:**
- Massive scale (96 layers, 12,288 d_model).
- Web‑scale training data (Common Crawl, WebText2, Books).
- Systematic study of scaling laws.

**Impact:** GPT‑3 was the first model to show that scaling alone could produce powerful task‑agnostic learning. It inspired the development of many subsequent language models and set the stage for instruction‑tuned and conversational models.

---

## 3. InstructGPT and ChatGPT: Alignment and Conversation (2022)

GPT‑3's base model did not reliably follow user instructions. Ouyang et al. addressed this with **InstructGPT**, which used **Reinforcement Learning from Human Feedback (RLHF)**:

1. **Supervised Fine‑Tuning (SFT):** on human‑written demonstrations.
2. **Reward Model (RM):** trained on human preference rankings.
3. **PPO (RLHF):** optimising the model against the reward model with a KL penalty.

The result was a model that was significantly more helpful, truthful, and harmless. A 1.3B InstructGPT was preferred by human evaluators over the 175B GPT‑3 base model.

In November 2022, OpenAI released **ChatGPT**, a conversational interface built on GPT‑3.5 with RLHF. It reached 100 million users in two months, sparking mainstream adoption of generative AI.

---

## 4. GPT‑4: Frontier Capability (2023)

The GPT‑4 technical report (2023) described a model that achieved human‑level performance on many professional and academic exams, with strong reasoning, coding, and safety capabilities. While architecture details were not disclosed, the model was believed to be a Transformer‑based mixture‑of‑experts, trained with extensive post‑training alignment.

**Key points:**
- Multimodal input capability (image + text) was described, though not initially deployed.
- Safety was a major focus, with adversarial testing and refusal mechanisms.
- The model set a new standard for practical LLM capability.

---

## 5. LLaMA and the Open‑Weight Ecosystem (2023)

Meta released **LLaMA** (Large Language Model Meta AI) as a research‑access model with sizes from 7B to 65B parameters. The paper demonstrated that smaller models trained on large quantities of public data could achieve competitive performance with GPT‑3.

**Contributions:**
- **Efficiency:** RMSNorm, SwiGLU, and careful training methodology.
- **Open‑Weight Access:** Catalysed a wave of derivative models (Alpaca, Vicuna, Llama‑2, Mistral).
- **Democratisation:** Enabled researchers to study, fine‑tune, and deploy LLMs without proprietary restrictions.

LLaMA's release accelerated the open‑source AI movement and demonstrated that open models could be a viable alternative to proprietary systems.

---

## 6. Beyond 2023: RAG, Tool Use, and Agentic AI

The story does not end with GPT‑4 and LLaMA. The 2020s have seen the emergence of:

- **RAG (Retrieval‑Augmented Generation):** combining LLMs with external knowledge retrieval.
- **Tool Use / Function Calling:** enabling LLMs to execute code, query APIs, and interact with external systems.
- **Agentic AI:** systems that can plan, reason, and act over extended timescales, with memory and governance.

These developments are turning LLMs from language models into **general‑purpose reasoning and action engines**.

---

## 7. Synthesis: From Biological Neurons to Agentic AI

The story of this project is the story of how a mathematical abstraction of a neuron (McCulloch & Pitts, 1943) evolved, through decades of incremental progress, into the foundation of modern artificial intelligence.

**The arc:**
- **1943–1958:** Biological foundations and the first trainable machine (Perceptron).
- **1960–1969:** Mathematics and optimisation (LMS, Perceptron limitations).
- **1974–1986:** Revival through backpropagation and neural networks.
- **1989–1997:** Architectural innovations (CNN, RNN, LSTM).
- **1998–2012:** Deep learning maturation (LeNet‑5, AlexNet).
- **2013–2017:** Modern architectures (Seq2Seq, GAN, ResNet, Transformer).
- **2018–2026:** The LLM era, scaling, alignment, open models, and agents.

---

## 8. Timeline of the LLM Era

| Year | Event | Significance |
| :--- | :--- | :--- |
| **2018** | GPT | Generative pre‑training and fine‑tuning |
| **2019** | GPT‑2 | Zero‑shot transfer via scaling |
| **2020** | GPT‑3 | Few‑shot learning at 175B scale |
| **2022** | InstructGPT | RLHF alignment |
| **2022** | ChatGPT | Conversational AI product milestone |
| **2023** | GPT‑4 | Capability leap and multimodal potential |
| **2023** | LLaMA | Open‑weight research catalyst |
| **2024–2026** | Agentic AI | Tools, memory, reasoning, and governance |