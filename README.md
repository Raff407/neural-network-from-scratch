# Neural Network From Scratch

A small neural network written by hand in Python with NumPy, including backpropagation, to understand how it actually works.

## The idea

A neural network is linear regression stacked in layers, with a non-linear function squeezed between them. Training is the same loop as linear regression, just with more steps in the middle.

1. **Forward pass**: push the inputs through each layer (multiply by weights, add bias, apply an activation such as ReLU or sigmoid) to get a prediction.
2. **Loss**: measure how wrong the prediction is.
3. **Backward pass (backpropagation)**: use the chain rule to work out, layer by layer from the end, how much each weight contributed to the error.
4. **Update**: nudge every weight against its gradient.
5. Repeat.

## Plan

- [ ] A network with one hidden layer that learns XOR, which a straight line cannot do
- [ ] Plot the loss falling over training
- [ ] Check one gradient numerically (nudge a weight slightly and see how the loss changes) to prove backprop is right
- [ ] *Optional:* classify handwritten digits (MNIST) and report test accuracy

## Data

XOR: four points you can type in yourself. MNIST, for the optional step, is on Kaggle as a CSV.

## Running

```bash
pip install -r requirements.txt
python network.py
```

## Status

Not started.
