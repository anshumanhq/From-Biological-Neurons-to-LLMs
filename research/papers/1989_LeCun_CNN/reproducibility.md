# Reproducibility: LeCun (1989)

**Paper:** Backpropagation Applied to Handwritten Zip Code Recognition  
**Dataset:** USPS Handwritten Digits (7291 training, 2007 test samples, 16×16 grayscale).  
**Hardware:** Sun workstation; training took several hours.  
**Initialisation:** Small uniform random weights (consistent with the period).  
**Activation:** Sigmoid / Tanh.  
**Pooling:** Average pooling (subsampling).  
**Optimiser:** SGD with fixed learning rate (no momentum, no Adam).  

---

## Reproducibility Today

The NumPy implementations in this archive replicate the core architectural principles:

- **Historical:** `implementation_historical.py` (sigmoid, average pooling, SGD, uniform init).
- **Modern:** `implementation_modern.py` (ReLU, max pooling, Adam, He init).

A full USPS‑level reproduction is beyond the scope of the scratch code, but the structural mechanics are fully demonstrated.

---

## How to Run

```bash
pip install numpy
python implementation_historical.py
python implementation_modern.py
```