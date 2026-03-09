# HMM-From-Scratch

**Author:** Edson Ricardo Pedroza Olivera  
**Version:** 1.0  
**Repository:** [https://github.com/epedrozao/HMM-From-Scratch](https://github.com/epedrozao/HMM-From-Scratch)

---

## Description

This library provides a complete and educational implementation of **Hidden Markov Models (HMM)** with discrete observations, written in pure Python with NumPy support. It includes the fundamental algorithms:

- **Viterbi** (decoding)
- **Forward** (evaluation)
- **Backward** (backward probability)
- **Forward-Backward** (smoothing)
- **Baum-Welch** (EM training)

Each algorithm is available in two versions:

- **Optimized**: uses logarithms and numerically stable techniques for working with long sequences without underflow.
- **Theoretical**: faithfully follows the textbook mathematical expressions, ideal for learning and understanding the algorithm.

The library is designed both for educational purposes and for use in real projects, and serves as a foundation for more specialized applications in **natural language processing**, **quantitative finance**, **bioinformatics**, and other fields.

---

## Features

- Clean, well-commented implementation with emphasis on readability and efficiency.
- Comprehensive parameter validation and edge case handling (empty sequences, unvisited states, etc.).
- Complete docstring documentation with executable examples (doctests).
- Optimized and theoretical versions of all algorithms.
- Compatible with Python 3.8+ and NumPy.

---

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/epedrozao/HMM-From-Scratch.git
cd HMM-From-Scratch
pip install .
```

Then import the class in your project:

```python
from hmm import HMM
```

> **Note:** NumPy is the only dependency and will be installed automatically.

---

## Quick Start

```python
import numpy as np
from hmm import HMM

# Define a simple model (2 states, 2 observations)
A  = np.array([[0.7, 0.3],
               [0.4, 0.6]])
B  = np.array([[0.8, 0.2],
               [0.4, 0.6]])
pi = np.array([0.8, 0.2])

hmm = HMM(2, 2, A=A, B=B, pi=pi)

# Observation sequence (0: happy, 1: grumpy)
obs = [0, 0, 1, 0]

# Decoding with Viterbi
path, prob = hmm.viterbi(obs)
print("Most probable states:", path)

# Log-likelihood with forward
_, log_lik = hmm.forward(obs, return_log=True)
print("Log-likelihood:", log_lik)

# Smoothing with forward-backward
gamma = hmm.forward_backward(obs)
print("Smoothed probabilities:\n", gamma)

# Parameter estimation with Baum-Welch (single sequence)
obs_train = [0, 0, 1, 0, 0, 1, 0, 0]
log_lik_final = hmm.baum_welch(obs_train, max_iter=50)
print("Final log-likelihood after training:", log_lik_final)

# Parameter estimation with Baum-Welch (multiple sequences)
obs_multi  = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])
lengths    = [4, 3, 3]
log_lik_multi = hmm.baum_welch_mult(obs_multi, lengths, max_iter=50)
print("Final log-likelihood (multi-sequence):", log_lik_multi)
```

---

## Method Reference

### Constructor

```python
HMM(n_estados, n_obs, A=None, B=None, pi=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_estados` | `int` | Number of hidden states N |
| `n_obs` | `int` | Number of possible observation symbols M |
| `A` | `ndarray`, optional | Transition matrix of shape (N, N). Each row must sum to 1. |
| `B` | `ndarray`, optional | Emission matrix of shape (N, M). Each row must sum to 1. |
| `pi` | `ndarray`, optional | Initial probability vector of shape (N,). Must sum to 1. |

If not provided, `A`, `B`, and `pi` are initialized randomly and normalized to be stochastic.

---

### Optimized Methods (numerically stable)

| Method | Description |
|--------|-------------|
| `viterbi(obs, return_log=False)` | Most probable hidden state sequence (Viterbi decoding) |
| `forward(obs, return_log=False)` | Forward matrix α and total likelihood P(O\|λ) |
| `backward(obs, return_log=False)` | Backward matrix β |
| `forward_backward(obs, return_log=False)` | Smoothed state probabilities γ |
| `baum_welch(obs, max_iter, tol, return_log_history)` | EM parameter estimation — single sequence |
| `baum_welch_mult(obs_concatenated, lengths, max_iter, tol, return_log_history)` | EM parameter estimation — multiple sequences |

---

#### `viterbi(obs, return_log=False)`

Finds the most probable hidden state sequence for an observation sequence using log-probabilities to prevent numerical underflow.

**Returns:** `path` (list of int), `best_log_prob` or `best_prob` (float)

---

#### `forward(obs, return_log=False)`

Computes the forward matrix α_t(j) = P(o_1, …, o_t, q_t = j | λ) and the total likelihood P(O | λ).

**Returns:**
- If `return_log=True`: `log_alpha` (ndarray, shape T×N), `log_likelihood` (float)
- If `return_log=False`: `alpha` (ndarray), `likelihood` (float)

---

#### `backward(obs, return_log=False)`

Computes the backward matrix β_t(j) = P(o_{t+1}, …, o_T | q_t = j, λ).

**Returns:**
- If `return_log=True`: `log_beta` (ndarray, shape T×N)
- If `return_log=False`: `beta` (ndarray)

---

#### `forward_backward(obs, return_log=False)`

Computes the smoothed probabilities γ_t(j) = P(q_t = j | O, λ).

**Returns:**
- If `return_log=True`: `log_gamma` (ndarray, shape T×N)
- If `return_log=False`: `gamma` (ndarray, rows sum to 1)

---

#### `baum_welch(obs, max_iter=100, tol=1e-6, return_log_history=False)`

Expectation-Maximization (EM) algorithm to estimate model parameters (π, A, B) from a single observation sequence. Iterates until the relative change in log-likelihood is smaller than `tol` or `max_iter` is reached.

**Returns:**
- If `return_log_history=False`: `log_likelihood` (float)
- If `return_log_history=True`: `log_likelihood` (float), `log_lik_history` (list of float)

---

#### `baum_welch_mult(obs_concatenated, lengths, max_iter=100, tol=1e-6, return_log_history=False)`

Multi-sequence extension of Baum-Welch. Accepts several independent observation sequences concatenated into a single 1-D array, with their individual lengths specified separately. Sufficient statistics (γ, ξ) are accumulated across all sequences in the E-step and parameters are re-estimated jointly in the M-step, which is equivalent to maximizing the total log-likelihood Σ_s log P(obs_s | model).

| Parameter | Type | Description |
|-----------|------|-------------|
| `obs_concatenated` | `array-like of int` | All sequences concatenated into a 1-D array. |
| `lengths` | `array-like of int` | Length of each individual sequence. Must sum to `len(obs_concatenated)`. |
| `max_iter` | `int` | Maximum number of EM iterations. |
| `tol` | `float` | Convergence tolerance on relative change in total log-likelihood. |
| `return_log_history` | `bool` | If `True`, also returns the per-iteration log-likelihood list. |

**Returns:**
- If `return_log_history=False`: `log_likelihood` (float) — total final log-likelihood Σ_s log P(obs_s | model)
- If `return_log_history=True`: `log_likelihood` (float), `log_lik_history` (list of float)

---

### Theoretical Methods (no logarithms, for learning)

These methods follow the textbook formulas exactly and are ideal for understanding how the algorithms work.

> **Warning:** these versions work in linear space and may suffer numerical underflow for long sequences. Use the optimized versions in production.

| Method | Returns |
|--------|---------|
| `viterbi_teorico(obs)` | `path` (list of int), `prob` (float) |
| `forward_teorico(obs)` | `alpha` (ndarray, T×N), `prob` (float) |
| `backward_teorico(obs)` | `beta` (ndarray, T×N) |
| `forward_backward_teorico(obs)` | `gamma` (ndarray, T×N) |
| `baum_welch_teorico(obs, max_iter, tol, return_log_history)` | same as `baum_welch` |

---

## Open Core Strategy

The core library (`hmm.py`) is published under the MIT License to encourage community use, study, and contribution. This allows:

- Demonstrating technical capability to recruiters and clients on platforms like Toptal or Gun.io.
- Building a reputation as a developer of high-quality tools.
- Serving as a foundation for more complex projects.

On top of this open core, specialized applications have been developed in two high-value commercial areas:

- **Natural language processing (NLP):** POS tagging, named entity recognition, HMM-based text generation.
- **Financial models:** market regime detection, time series forecasting, risk analysis.

These applications are distributed under a commercial license and are available to companies and professionals who wish to integrate them into their products or services. If you are interested, contact the author to discuss the terms.

This Open Core model ensures that the fundamental tool remains free and accessible, while specific business solutions are commercialized fairly, protecting intellectual property and enabling continued development.

---

## License

- **Core library (`hmm.py`):** MIT License. You may use, copy, modify, and distribute this code freely, provided the copyright notice is retained.
- **Specialized applications (NLP, finance):** Commercial license. For more information, contact the author.

---

## Contact

- **Email:** epedroz2@itam.mx · edsonpedroza99@gmail.com
- **LinkedIn:** [Edson Ricardo Pedroza Olivera](https://www.linkedin.com/in/edson-pedroza-b0108232a)
- **GitHub:** [@epedrozao](https://github.com/epedrozao)

For questions, suggestions, or interest in commercial applications, feel free to reach out.

---

## References

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proceedings of the IEEE.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
- Wikipedia: [Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)

---

*Enjoy exploring the world of HMMs!*
