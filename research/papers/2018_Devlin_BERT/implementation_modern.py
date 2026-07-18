"""
BERT (2018) – Modern Educational Implementation

This is a conceptual demonstration of BERT's core innovations:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Bidirectional Transformer Encoder

Reference: Devlin et al., 2018
"""

import numpy as np

def mlm_demo():
    print("=== Masked Language Modeling (MLM) ===")
    print("Input: The cat sat on the [MASK].")
    print("Output: The cat sat on the mat.")
    print("Tokens attend to both left and right context.")
    print("15% of tokens are masked during pre-training.\n")

def nsp_demo():
    print("=== Next Sentence Prediction (NSP) ===")
    print("A: The cat sat on the mat.")
    print("B: The dog sat on the rug.")
    print("→ isNext: 1 (yes)")
    print("A: The cat sat on the mat.")
    print("B: It was a sunny day.")
    print("→ isNext: 0 (no)\n")

def bidirectional_demo():
    print("=== Bidirectional Context ===")
    print("BERT uses bidirectional self-attention.")
    print("Each token can attend to every other token.")
    print("Contrast with GPT (decoder-only, causal masking).\n")

def fine_tuning_demo():
    print("=== Pre-training + Fine-tuning ===")
    print("Pre-training objectives: MLM + NSP")
    print("Pre-training data: BooksCorpus (800M words) + Wikipedia (2.5B words)")
    print("Fine-tuned on: GLUE, SQuAD, SWAG, etc.\n")

if __name__ == "__main__":
    print("=== BERT 2018 Modern Demo ===")
    mlm_demo()
    nsp_demo()
    bidirectional_demo()
    fine_tuning_demo()