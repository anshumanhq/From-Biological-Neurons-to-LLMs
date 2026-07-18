"""
BERT (2018) – Conceptual Implementation Demonstration
BERT uses the Transformer encoder with bidirectional self-attention.
Key contributions: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

Note: This is a conceptual demonstration only.
"""

import numpy as np

def masked_language_modeling_demo():
    """Demonstrate the MLM objective."""
    print("=== Masked Language Modeling (MLM) ===")
    print("Input: [CLS] The [MASK] is a domestic animal.")
    print("Task: Predict the masked token: 'cat'")
    print("The model uses both left and right context: 'The' and 'is a domestic animal'")
    print("This enables deep bidirectional representation.\n")

def next_sentence_prediction_demo():
    """Demonstrate the NSP objective."""
    print("=== Next Sentence Prediction (NSP) ===")
    print("Input: [CLS] Sentence A [SEP] Sentence B [SEP]")
    print("Task: Predict whether Sentence B follows Sentence A.")
    print("This objective helps the model understand relationships between sentences.\n")

def pre_training_fine_tuning_demo():
    """Demonstrate the pre-training + fine-tuning paradigm."""
    print("=== Pre-training + Fine-tuning ===")
    print("Pre-training: On large corpora (BooksCorpus, Wikipedia) with MLM + NSP.")
    print("Fine-tuning: On specific tasks (GLUE, SQuAD) with task-specific heads.")
    print("This paradigm became the standard for NLP.\n")

def architecture_demo():
    """Demonstrate the architecture."""
    print("=== BERT Architecture ===")
    print("Encoder-only Transformer (no decoder).")
    print("Base model: 12 layers, 768 hidden, 12 heads, 110M parameters.")
    print("Large model: 24 layers, 1024 hidden, 16 heads, 340M parameters.\n")

class BERT_2018:
    def __init__(self):
        print("BERT (2018) — Conceptual Demonstration")
        print("Bidirectional Transformer encoder with MLM + NSP.\n")

    def forward(self):
        architecture_demo()
        masked_language_modeling_demo()
        next_sentence_prediction_demo()
        pre_training_fine_tuning_demo()


if __name__ == "__main__":
    print("=== BERT 2018 Conceptual Demo ===")
    bert = BERT_2018()
    bert.forward()
    print("\nNote: This is a conceptual demonstration of BERT's core ideas.")