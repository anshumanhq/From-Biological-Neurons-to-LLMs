# Summary: Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Year:** 2017

The Transformer introduced a sequence transduction architecture based solely on attention mechanisms, eliminating recurrence and convolution. Scaled dot-product attention \( \text{softmax}(QK^T/\sqrt{d_k})V \) and multi-head attention allow parallel processing, enabling efficient training and state-of-the-art results on machine translation. This architecture became the foundation for all modern large language models.