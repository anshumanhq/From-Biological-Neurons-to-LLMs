# Summary: Mixtral of Experts

**Authors:** Albert Q. Jiang et al.  
**Year:** 2024

Mixtral 8×7B introduced a sparse Mixture of Experts (MoE) architecture with 8 experts (each ~7B parameters) and top‑2 routing. The model has a large total parameter pool (~47B) but uses only a sparse subset of experts per token, achieving strong performance while maintaining inference efficiency comparable to a dense 7B model. Mixtral outperforms Llama 2 70B and GPT‑3.5 on several benchmarks, establishing sparse MoE as a viable direction for efficient scaling.