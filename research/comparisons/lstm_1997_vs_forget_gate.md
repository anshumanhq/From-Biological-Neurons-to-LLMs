# Comparison: LSTM 1997 vs. LSTM with Forget Gate (1999/2000)

## Overview

The original 1997 LSTM (Hochreiter & Schmidhuber) had no mechanism to reset its internal state. The forget gate was introduced to allow the network to adaptively forget when processing continuous streams.

| Aspect | 1997 LSTM | 1999/2000 LSTM (with forget gate) |
| :--- | :--- | :--- |
| **Cell state update** | \( c_t = c_{t-1} + i_t \odot \tilde{c}_t \) | \( c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \) |
| **Forget gate** | No | Yes (\( f_t \)) |
| **Gradient flow** | \(\partial c_t/\partial c_{t-1} = 1\) (idealised) | \(\partial c_t/\partial c_{t-1} = f_t\) (adaptive) |
| **Memory reset** | Not possible (CEC always retains) | Possible via \( f_t \to 0 \) |
| **Use case** | Memory retention over fixed intervals | Continuous input streams (e.g., speech) |

## Why the Forget Gate Matters

- Allows the network to **forget** irrelevant information, preventing state saturation.
- Enables **online** learning where the context changes over time.
- Becomes the standard LSTM taught today.

## Legacy

Modern LSTM implementations always include the forget gate. Many researchers incorrectly attribute it to the 1997 paper.