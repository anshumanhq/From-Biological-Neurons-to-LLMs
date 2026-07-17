# Summary: Attractor Dynamics and Parallelism in a Connectionist Sequential Machine

**Authors:** Michael I. Jordan  
**Year:** 1990

The Jordan Network is the first practical recurrent neural network for sequence learning. It introduces **context units** that store the previous output and feed it back to the hidden layer, giving the network a form of short-term memory. The network is trained using **backpropagation through time (BPTT)** with **teacher forcing**, where the context is updated using the target output during training to stabilise learning.

This work established the foundation for all subsequent sequence models, including Elman Networks, LSTMs, and Transformers.