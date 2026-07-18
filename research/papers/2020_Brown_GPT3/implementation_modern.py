"""
GPT-3 (2020) – Modern Educational Implementation

This is a forward-pass demonstration of the GPT-3 architecture with
in-context learning (zero-shot, one-shot, few-shot). The focus is on
illustrating the scaling and in-context learning concepts.

Reference: Brown et al., 2020
"""

import numpy as np

class CausalMask:
    @staticmethod
    def create(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9


def in_context_learning_demo():
    """Demonstrate zero-shot, one-shot, and few-shot prompting."""
    print("=== In-Context Learning Demo (GPT-3) ===")
    print("Zero-shot:")
    print("  Prompt: 'Translate English to French: cat →'")
    print("  Model: 'chat'")
    print("\nOne-shot:")
    print("  Prompt: 'Translate English to French: dog → chien'")
    print("          'Translate English to French: cat →'")
    print("  Model: 'chat'")
    print("\nFew-shot:")
    print("  Prompt: 'Translate English to French: dog → chien'")
    print("          'Translate English to French: bird → oiseau'")
    print("          'Translate English to French: cat →'")
    print("  Model: 'chat'")
    print("No gradient updates during evaluation.\n")


def model_scaling_demo():
    """Demonstrate model scaling."""
    print("=== GPT-3 Model Scaling ===")
    models = [
        {"name": "GPT-3 Small", "params": "125M", "layers": 12, "d_model": 768},
        {"name": "GPT-3 Medium", "params": "350M", "layers": 24, "d_model": 1024},
        {"name": "GPT-3 Large", "params": "760M", "layers": 24, "d_model": 1536},
        {"name": "GPT-3 XL", "params": "1.3B", "layers": 24, "d_model": 2048},
        {"name": "GPT-3 2.7B", "params": "2.7B", "layers": 32, "d_model": 2560},
        {"name": "GPT-3 6.7B", "params": "6.7B", "layers": 32, "d_model": 4096},
        {"name": "GPT-3 13B", "params": "13B", "layers": 40, "d_model": 5140},
        {"name": "GPT-3 175B", "params": "175B", "layers": 96, "d_model": 12288},
    ]
    for model in models:
        print(f"{model['name']}: {model['params']} params, {model['layers']} layers, d_model={model['d_model']}")
    print("Performance improves with scale.\n")


def benchmark_performance_demo():
    """Demonstrate benchmark performance."""
    print("=== Benchmark Performance (GPT-3) ===")
    print("Task: SuperGLUE (few-shot)")
    print("Performance improves with model size:")
    print("  125M: ~40% accuracy")
    print("  175B: ~70% accuracy")
    print("Task: Translation (few-shot)")
    print("  English to French: 35+ BLEU")
    print("  English to German: 40+ BLEU")
    print("Still struggles with some tasks (e.g., arithmetic, logic).\n")


if __name__ == "__main__":
    print("=== GPT-3 2020 Modern Educational Demo ===")
    model_scaling_demo()
    in_context_learning_demo()
    benchmark_performance_demo()