# Experiments

Standalone scripts that reproduce the figures in the project README. Each script
accepts a `--seed` argument and saves its output to `../figures/`.

Run any script with:

```bash
uv run python experiments/<script>.py --seed 0
```

## Planned scripts

- `xor.py` — solving XOR with one hidden layer
- `moons_decision_boundary.py` — decision boundary and loss curves
- `mnist.py` — full MNIST training run and error analysis
- `optimizer_comparison.py` — SGD, momentum and Adam on one plot
- `initialisation_study.py` — zero, small random and He initialisation
