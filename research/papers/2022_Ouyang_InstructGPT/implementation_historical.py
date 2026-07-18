"""
InstructGPT (2022) – Historical Concept Demonstration
Pipeline:
- Supervised Fine-Tuning (SFT) on human demonstrations
- Reward Model (RM) trained on human preference rankings
- RLHF / PPO optimization against the reward model

Note: This is a conceptual demonstration of the RLHF pipeline.
Full-scale training on the GPT-3 175B model is beyond the scope.
"""

import numpy as np
from typing import List, Tuple

class SupervisedFineTuning:
    """Conceptual SFT on human-written demonstrations."""
    def __init__(self, model):
        self.model = model

    def train(self, demonstrations: List[Tuple[str, str]]):
        """
        demonstrations: list of (prompt, response) pairs
        """
        print("=== SFT Phase ===")
        print(f"Training on {len(demonstrations)} human-written demonstrations.")
        print("Model learns to mimic human responses.\n")
        # Simulated training
        return self.model

class RewardModel:
    """Conceptual reward model trained on preference rankings."""
    def __init__(self):
        self.weights = np.random.randn(1, 1) * 0.1

    def train(self, preference_data: List[Tuple[str, str, str]]):
        """
        preference_data: list of (prompt, response_w, response_l)
        """
        print("=== Reward Model Phase ===")
        print(f"Training on {len(preference_data)} human preference comparisons.")
        print("Model learns to assign higher scores to preferred responses.\n")
        # Simulated training
        return self

    def predict(self, prompt: str, response: str) -> float:
        """Return a reward score for a (prompt, response) pair."""
        # Simulated reward score
        return float(np.dot(np.random.randn(1, 1), np.random.randn(1, 1)))

class PPOOptimizer:
    """Conceptual PPO-based RLHF optimization."""
    def __init__(self, model, reward_model, ref_model, kl_penalty=0.01):
        self.model = model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.kl_penalty = kl_penalty

    def optimize(self, prompts: List[str], epochs: int = 10):
        """Optimise the model against the reward model with KL penalty."""
        print("=== PPO / RLHF Phase ===")
        print(f"Optimizing on {len(prompts)} prompts for {epochs} epochs.")
        print("PPO objective: maximize reward - β * KL(π || π_ref)")
        print("The model learns to generate responses that are highly rewarded.\n")
        return self.model


class InstructGPT_2022:
    """
    InstructGPT pipeline (Ouyang et al., 2022).
    Demonstrates the SFT → RM → PPO / RLHF alignment pipeline.
    """
    def __init__(self, model_name="GPT-3-175B"):
        self.model_name = model_name
        print(f"InstructGPT pipeline initialized with {model_name}.")

    def sft(self, demonstrations):
        """Supervised fine-tuning phase."""
        sft = SupervisedFineTuning(self)
        return sft.train(demonstrations)

    def reward_modeling(self, preference_data):
        """Reward model training phase."""
        rm = RewardModel()
        return rm.train(preference_data)

    def ppo(self, prompts, rm, ref_model, kl_penalty=0.01):
        """PPO / RLHF phase."""
        optimizer = PPOOptimizer(self, rm, ref_model, kl_penalty)
        return optimizer.optimize(prompts)


if __name__ == "__main__":
    print("=== InstructGPT 2022 Pipeline Demo ===")
    print("Note: This is a conceptual demonstration of the RLHF pipeline.\n")
    instructgpt = InstructGPT_2022("GPT-3-175B")
    demonstrations = [
        ("Translate French to English: 'chat'", "cat"),
        ("Q: What is 2+2? A:", "4"),
    ]
    preference_data = [
        ("Translate French to English: 'chien'", "dog", "dog? no"),
        ("Q: What is 5+5? A:", "10", "maybe 10?"),
    ]
    prompts = ["Q: What is 10+10? A:", "Summarize the text: ..."]
    ref_model = None

    instructgpt.sft(demonstrations)
    rm = instructgpt.reward_modeling(preference_data)
    instructgpt.ppo(prompts, rm, ref_model)
    print("InstructGPT pipeline demonstrated.")