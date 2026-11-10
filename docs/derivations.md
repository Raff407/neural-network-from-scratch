# Derivations — Neural Network From Scratch

This document records the mathematical derivations underpinning the implementation,
written out in full before the corresponding code.

## Summary of results

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

## Stage 1: Logistic regression

_To be completed._

## Stage 2: One hidden layer

_To be completed._

## Stage 3: Modular library

_To be completed._

## Stage 4: Training loop

_To be completed._

## Stage 5: MNIST

_To be completed._

## Stage 6: Optimisers and regularisation

_To be completed._

## Stage 7: Cross-check and write-up

_To be completed._
