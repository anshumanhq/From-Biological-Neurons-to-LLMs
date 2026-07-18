# Summary: Training Language Models to Follow Instructions with Human Feedback

**Authors:** Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe  
**Year:** 2022

InstructGPT demonstrated that fine-tuning a pretrained language model with Reinforcement Learning from Human Feedback (RLHF) could significantly improve instruction following and alignment with human preferences. The pipeline involved supervised fine-tuning on human-written demonstrations, reward model training on human preference rankings, and PPO-based RLHF optimization. Human evaluators preferred outputs from the 1.3B InstructGPT model over the 175B GPT-3 base model on the evaluated prompt distribution.