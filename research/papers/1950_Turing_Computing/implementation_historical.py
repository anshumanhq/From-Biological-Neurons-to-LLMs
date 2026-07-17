'''
Turing (1950) – Child Machine Teaching Simulation
This is a conceptual simulation of Turing's "reward and punishment" 
education system, comparing it to Hebbian learning.
'''

import numpy as np

def turing_child_learning(weights, input_vector, target_response, reward_scale=0.01, punish_scale=0.01):
    """
    Simulates Turing's 'Child Machine' education concept.
    If the output matches the target (reward), we strengthen active pathways.
    If the output mismatches (punishment), we weaken active pathways.
    This is a simple associative reinforcement mechanism.
    """
    output = np.dot(weights, input_vector)
    # Simulate a binary decision
    decision = 1 if output > 0 else 0

    if decision == target_response:
        # Reward: Hebbian-style strengthening (Turing's reinforcement)
        delta_w = reward_scale * np.outer(input_vector, output)
        weights += delta_w
        print("[Reward] Strengthened active synapses.")
    else:
        # Punishment: Weakening active synapses (inhibitory Hebbian)
        delta_w = punish_scale * np.outer(input_vector, output)
        weights -= delta_w
        print("[Punishment] Weakened active synapses.")

    return weights, decision

# Demonstration
if __name__ == "__main__":
    print("=== Turing Child Machine: Teaching a Response ===")
    W = np.array([[0.2, 0.1]])  # immature initial weights
    x = np.array([1.0, 0.0])    # stimulus (e.g., "show me a circle")
    target = 1                  # desired response (e.g., "yes")

    for epoch in range(5):
        W, dec = turing_child_learning(W, x, target)
        print(f"Weights: {W}, Decision: {dec}\n")

    print("=== Comparison with Hebbian Learning ===")
    # Hebbian rule (from 1949)
    def hebbian_update(W, pre, post, lr=0.01):
        return W + lr * np.outer(pre, post)

    W_hebb = np.array([[0.2, 0.1]])
    pre = np.array([1.0, 0.0])
    post = np.array([1.0])  # post fires
    print(f"Hebbian update result: {hebbian_update(W_hebb, pre, post)}")
    print("Conclusion: Turing's reward mechanism is functionally Hebbian, but contextualized within a teacher-student framework.")