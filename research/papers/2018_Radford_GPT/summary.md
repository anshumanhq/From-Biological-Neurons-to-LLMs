# Summary: Improving Language Understanding by Generative Pre-Training

**Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever  
**Year:** 2018

GPT introduced generative pre-training of a Transformer decoder on a large text corpus (BooksCorpus, ~5GB), followed by supervised fine-tuning on downstream tasks. The model uses a 12-layer Transformer decoder with masked self-attention (causal masking) to perform autoregressive language modelling. It established the "pre-train then fine-tune" paradigm that became the foundation for modern LLMs.