# Vanishing Gradient

A problem in deep networks where gradients become exponentially small as they propagate backward, making it difficult to update weights in early layers. This is especially severe in RNNs trained with BPTT on long sequences. LSTM and GRU were designed to mitigate this.
