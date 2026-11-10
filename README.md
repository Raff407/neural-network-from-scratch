# Neural Network From Scratch

[![CI](https://github.com/Raff407/neural-network-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Raff407/neural-network-from-scratch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-in%20development-orange)

A minimal deep-learning library written in NumPy, implementing layers, losses, backpropagation and optimisers without any framework, verified with numerical gradient checks and trained on XOR, two moons and MNIST.

> **Status:** in development. The repository structure, tooling and test infrastructure are in place; implementation is proceeding stage by stage according to the roadmap below.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Mathematical background](#mathematical-background)
- [Roadmap](#roadmap)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Experiments](#experiments)
- [Design principles](#design-principles)
- [References](#references)
- [License](#license)

## Overview

This project builds a feed-forward neural network library from first principles. Each layer exposes `forward` and `backward`; a `Sequential` container composes them; losses and optimisers are implemented as independent components. Every analytic gradient is verified against a central finite-difference estimate before any training is performed.

The library is exercised on problems of increasing difficulty: logistic regression on separable data, XOR (which a linear model cannot solve), the two-moons dataset, and finally MNIST with a target test accuracy of at least 97%. Results are cross-checked against an equivalent PyTorch model.

## Features

- Layers: `Dense`, `ReLU`, `Sigmoid`, `Tanh`
- Losses: MSE, binary cross-entropy and a numerically stable fused softmax cross-entropy
- `Sequential` model with parameter access and prediction
- Optimisers: SGD, momentum and Adam, returning new parameter arrays
- He and Xavier initialisation; L2 weight decay and dropout
- Mini-batch training loop with shuffling, metrics and loss curves
- Finite-difference gradient checking for every layer and loss

## Mathematical background

For a layer with input $a \in \mathbb{R}^{B \times m}$, weights $W \in \mathbb{R}^{m \times k}$ and bias $b$:

| Component | Definition |
|---|---|
| Forward pass | $z = aW + b,\quad a' = f(z)$ |
| Softmax | $p_i = \exp(z_i - \max z) / \sum_j \exp(z_j - \max z)$ |
| Cross-entropy | $L = -\frac{1}{B} \sum \log p_{y}$ |
| Softmax + CE gradient | $\partial L / \partial z = (p - y_{\text{onehot}}) / B$ |
| Backpropagation | $\delta_{\text{prev}} = (\delta W^\top) \odot f'(z_{\text{prev}})$ |
| Parameter gradients | $\partial L / \partial W = a^\top \delta,\quad \partial L / \partial b = \sum_{\text{batch}} \delta$ |
| Adam | $m \leftarrow \beta_1 m + (1-\beta_1) g,\ v \leftarrow \beta_2 v + (1-\beta_2) g^2,\ w \leftarrow w - \alpha \hat{m} / (\sqrt{\hat{v}} + \epsilon)$ |
| Gradient check | $\frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon}$, relative error $< 10^{-7}$ |

Full derivations are recorded in [`docs/derivations.md`](docs/derivations.md).

## Roadmap

- [ ] **Stage 1: Logistic regression.** Single sigmoid neuron with binary cross-entropy; gradient check passes.
- [ ] **Stage 2: One hidden layer.** Manual backpropagation for a 2-4-1 network that solves XOR.
- [ ] **Stage 3: Modular library.** Layer abstraction, fused softmax cross-entropy and `Sequential`; per-layer gradient checks.
- [ ] **Stage 4: Training loop.** Mini-batching, shuffling, metrics; decision boundary on two moons.
- [ ] **Stage 5: MNIST.** 784-128-64-10 ReLU network reaching ≥ 97% test accuracy; confusion matrix and error analysis.
- [ ] **Stage 6: Optimisers and regularisation.** SGD vs momentum vs Adam; initialisation study; weight decay, dropout, batch normalisation.
- [ ] **Stage 7: Cross-check and write-up.** Comparison with an equivalent PyTorch model; ≥ 80% coverage.

## Project structure

```text
neural-network-from-scratch/
├── .github/workflows/ci.yml   # Lint, format check and tests on Python 3.12 / 3.13
├── data/                      # Downloaded datasets (git-ignored)
├── docs/derivations.md        # Mathematical derivations
├── experiments/               # Reproducible scripts that generate figures
├── figures/                   # Saved plots referenced in this README
├── src/
│   └── neural_net/
│       ├── __init__.py    # Package metadata
│       ├── layers.py      # Dense layer and activation layers (ReLU, Sigmoid, Tanh)
│       ├── losses.py      # MSE, binary cross-entropy and fused softmax cross-entropy
│       ├── model.py       # Sequential container composing layers
│       ├── optimizers.py  # SGD, momentum and Adam optimisers
│       ├── init.py        # He and Xavier weight initialisation
│       ├── train.py       # Mini-batch iteration, training loop and metrics
│       ├── gradcheck.py   # Finite-difference gradient checking
│       └── data.py        # Loaders for XOR, two moons and MNIST
├── tests/                     # pytest suite
├── pyproject.toml             # Packaging and tool configuration
└── uv.lock                    # Locked dependency versions
```

## Getting started

### Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/Raff407/neural-network-from-scratch.git
cd neural-network-from-scratch
uv sync
```

### Running the tests

```bash
uv run pytest
```

### Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
```

## Experiments

Each script in [`experiments/`](experiments/) is deterministic given its seed and writes its output to [`figures/`](figures/).

- `xor.py` — solving XOR with one hidden layer
- `moons_decision_boundary.py` — decision boundary and loss curves
- `mnist.py` — full MNIST training run and error analysis
- `optimizer_comparison.py` — SGD, momentum and Adam on one plot
- `initialisation_study.py` — zero, small random and He initialisation

Results and figures will be added to this section as each stage is completed.

## Design principles

- **From scratch.** The algorithm itself uses only Python and NumPy. scikit-learn (and, where noted, PyTorch) is used solely as a reference implementation in tests and for loading datasets.
- **Test-driven.** Each stage begins with a failing test against a hand-computed value or a reference implementation. Target coverage is at least 80%.
- **No mutation.** Public functions return new arrays and objects and never modify their inputs.
- **Reproducible.** Every stochastic routine and experiment accepts an explicit random seed.
- **Validated boundaries.** Public functions check input shapes and dtypes and fail with clear error messages.

## References

1. M. Nielsen — *Neural Networks and Deep Learning*, ch. 1–2. <http://neuralnetworksanddeeplearning.com>
2. A. Karpathy — *Neural Networks: Zero to Hero* (micrograd). <https://karpathy.ai/zero-to-hero.html>
3. Stanford CS231n — Course notes on backpropagation and training. <https://cs231n.github.io>
4. I. Goodfellow, Y. Bengio, A. Courville — *Deep Learning*, ch. 6 and 8. <https://www.deeplearningbook.org>
5. D. P. Kingma, J. Ba — *Adam: A Method for Stochastic Optimization* (2014). <https://arxiv.org/abs/1412.6980>

## License

Released under the [MIT License](LICENSE).
