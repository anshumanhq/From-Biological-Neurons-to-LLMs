# Historical Narrative: Modern Neural Networks – Seq2Seq, GAN, ResNet, Transformer (2013–2017)

**Author:** Anshuman Singh  
**Project:** From Biological Neurons to Large Language Models  
**Archive Status:** Milestone 6 – Complete (Seq2Seq 2014, GAN 2014, ResNet 2015, Transformer 2017)  
**Last Updated:** 2026-07-18

---

## Introduction

The period from 2013 to 2017 was one of explosive expansion. Deep learning, validated by AlexNet, moved beyond vision into sequence modelling, generative modelling, and the fundamental architecture of neural networks itself. This era produced four foundational innovations:

1. **Seq2Seq (2014)** – the encoder‑decoder architecture for neural machine translation.
2. **GAN (2014)** – adversarial training for generative models.
3. **ResNet (2015)** – residual connections enabling extremely deep networks.
4. **Transformer (2017)** – self‑attention replacing recurrence, becoming the architecture of choice for sequence transduction.

Together, these works set the stage for the foundation models that dominate the 2020s.

---

## 1. Seq2Seq: Neural Machine Translation (2014)

Sutskever, Vinyals, and Le introduced the **sequence‑to‑sequence** (Seq2Seq) architecture for neural machine translation. The model used two LSTMs:

- **Encoder:** processes the source sequence into a fixed‑length context vector.
- **Decoder:** generates the target sequence from the context vector.

**Key innovations:**
- **Teacher forcing:** using target tokens as inputs to the decoder during training.
- **Reversing source sequences:** improved performance by creating shorter‑term dependencies.

**Impact:** Seq2Seq became the standard for neural machine translation and established the encoder‑decoder paradigm, which would later be extended with attention and Transformers.

---

## 2. GAN: Generative Adversarial Networks (2014)

Ian Goodfellow et al. introduced the **Generative Adversarial Network**, a framework for training generative models through competition:

- **Generator \( G \):** maps latent noise to generated samples.
- **Discriminator \( D \):** classifies samples as real or fake.

The training objective is a minimax game:

'''latex
\min_G \max_D V(D,G) =
\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
+
\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
'''

**Significance:** GANs provided a powerful alternative to traditional generative modelling, enabling high‑quality image generation and inspiring a vast body of subsequent work (DCGAN, WGAN, StyleGAN).

---

## 3. ResNet: Residual Learning (2015)

He et al. addressed the **degradation problem**: as networks became deeper, training error increased despite the availability of more capacity. Their solution was **residual learning**: instead of learning the desired mapping \( \mathcal{H}(x) \), learn the residual \( \mathcal{F}(x) = \mathcal{H}(x) - x \), and add an identity shortcut:

'''latex
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}
'''

**Why it worked:**
- Identity shortcuts provide a direct path for gradients, enabling training of networks with >100 layers.
- Residual connections became a universal component of deep architectures.

**Impact:** ResNet won the 2015 ILSVRC with 3.57% top‑5 error, surpassing human performance. Its design principle—adding shortcuts to ease optimisation—was adopted by Transformers, DenseNet, and many subsequent models.

---

## 4. Transformer: Attention Is All You Need (2017)

Vaswani et al. proposed the **Transformer**, a sequence transduction architecture that eliminated recurrence and convolution entirely, relying instead on **self‑attention**.

**Core components:**
- **Scaled dot‑product attention:** \( \text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V \)
- **Multi‑head attention:** concatenating multiple attention heads to capture different relationships.
- **Positional encoding:** sin/cos functions to inject position information.
- **Encoder‑decoder structure** with residual connections and layer normalisation.

**Why it succeeded:**
- Parallel processing: all positions are attended to simultaneously, enabling efficient training.
- Long‑range dependencies: self‑attention captures relationships across the entire sequence.
- State‑of‑the‑art performance on WMT 2014 translation tasks.

**Impact:** The Transformer became the foundation of modern NLP. It replaced RNNs in almost all applications and gave rise to GPT, BERT, and the entire ecosystem of large language models.

---

## 5. Synthesis: From Vision to Language

The period 2013–2017 saw deep learning expand from vision to language, generation, and general‑purpose architectures. The Transformer, in particular, was the most consequential innovation since backpropagation; it set the stage for the LLM era, where scale and data would replace architectural complexity.

---

## 6. Bridge to the Next Era (2018–2026)

The Transformer was immediately adopted for language modelling. GPT (2018) and BERT (2018) demonstrated its power on language understanding tasks, and scaling soon revealed emergent capabilities that would transform AI. The narrative continues with *The LLM Era (2018–2026)*.

---

## Timeline of the Modern Neural Networks Era

| Year | Event | Significance |
| :--- | :--- | :--- |
| **2014** | Seq2Seq | Encoder‑decoder for neural translation |
| **2014** | GAN | Adversarial training for generative models |
| **2015** | ResNet | Residual connections for deep networks |
| **2017** | Transformer | Self‑attention replaces recurrence |