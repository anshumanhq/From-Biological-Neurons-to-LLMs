"""
GPT-2 (2019) – Modern Educational Implementation

This is a forward-pass demonstration of the GPT-2 architecture with
causal (masked) self-attention. The focus is on illustrating the
decoder-only Transformer structure and zero-shot transfer concept.

Reference: Radford et al., 2019
"""

import numpy as np

class CausalMask:
    @staticmethod
    def create(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9


def causal_attention_demo():
    """Demonstrate causal masking in self-attention."""
    print("=== Causal Masking Demo (GPT-2) ===")
    seq_len = 5
    d_k = 4
    Q = np.random.randn(seq_len, d_k)
    K = Q.copy()
    V = Q.copy()
    mask = CausalMask.create(seq_len)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    masked_scores = scores + mask
    attn_weights = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    print(f"Sequence length: {seq_len}")
    print("Causal mask (upper triangular -inf):")
    print(np.where(mask == -1e9, " -inf", "    0"))
    print("Note: Each position can only attend to itself and previous positions.\n")


def model_variants_demo():
    """Demonstrate GPT-2 model variants."""
    print("=== GPT-2 Model Variants ===")
    variants = {
        "Small": {"params": "117M", "layers": 12, "d_model": 768, "heads": 12},
        "Medium": {"params": "345M", "layers": 24, "d_model": 1024, "heads": 16},
        "Large": {"params": "774M", "layers": 36, "d_model": 1280, "heads": 20},
        "XL": {"params": "1.5B", "layers": 48, "d_model": 1600, "heads": 25},
    }
    for name, details in variants.items():
        print(f"{name}: {details['params']} params, {details['layers']} layers, "
              f"d_model={details['d_model']}, heads={details['heads']}")
    print("The XL (1.5B) model was the headline model.\n")


def zero_shot_transfer_demo():
    """Conceptual demonstration of zero-shot task transfer."""
    print("=== Zero-shot Transfer Demo ===")
    print("GPT-2 demonstrated that language models can perform tasks")
    print("without supervised fine-tuning by conditioning on a prompt:")
    print("  Prompt: 'Translate English to French: cat → '")
    print("  Model: 'chat'")
    print("  Prompt: 'Q: What is 2+2? A:'")
    print("  Model: '4'")
    print("This is zero-shot transfer: tasks are specified by the prompt alone.\n")


if __name__ == "__main__":
    print("=== GPT-2 2019 Modern Educational Demo ===")
    print("This demonstrates the core components of GPT-2.\n")
    model_variants_demo()
    causal_attention_demo()
    zero_shot_transfer_demo()
    print("GPT-2 components demonstrated.")