# Summary: Deep Residual Learning for Image Recognition

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Year:** 2015 (arXiv) / 2016 (CVPR)

ResNet introduced residual learning with identity shortcut connections to address the degradation problem in very deep networks. The formulation \( \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x} \) allows gradients to flow directly through the network, enabling training of networks with 152 layers. It won the ILSVRC 2015 competition with 3.57% top‑5 error, surpassing human‑level performance for the first time. Residual connections have since become a fundamental component of many architectures, including Transformers.