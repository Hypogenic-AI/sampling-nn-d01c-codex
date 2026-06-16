# Code Walkthrough

## Overview

The experiment is implemented in `src/sampling_nn_experiment.py`. It is intentionally a plain Python runner rather than a notebook so the whole pipeline can be reproduced with one command.

```bash
source .venv/bin/activate
python src/sampling_nn_experiment.py
```

## Data Flow

Local HuggingFace datasets are loaded from `datasets/{mnist,fashion_mnist}`. For each dataset and seed, the runner:

1. Creates stratified train/validation subsets from the original train split.
2. Uses the official test split for evaluation.
3. Computes normalization mean/std on the training subset only.
4. Converts PIL images to normalized PyTorch tensors.
5. Wraps tensors in deterministic `DataLoader` objects.

Metadata is written to `results/data_metadata.csv`.

## Model

`SmallCNN` has two convolutional blocks, a 128-dimensional penultimate hidden vector, optional dropout, and a 10-way classifier head. The forward method accepts a `hidden_intervention` callable so sampling can be inserted at the same internal location for every method.

## Interventions

- `deterministic`: no intervention.
- `output_sample`: samples class labels from final softmax probabilities and averages one-hot samples.
- `hidden_categorical`: samples hidden dimensions from `softmax(abs(h) / tau)` and rescales sampled dimensions by inverse probability.
- `sap`: samples a Bernoulli hidden mask with probabilities proportional to absolute activation magnitude.
- `gaussian`: adds activation noise proportional to each example's hidden standard deviation.
- `mc_dropout`: keeps dropout active during inference for the dropout-trained model.

## Metrics

`compute_metrics` reports accuracy, negative log likelihood, expected calibration error, Brier score, predictive entropy, probability variance across Monte Carlo samples, and per-class accuracy summaries.

Raw rows are written to `results/metrics_raw.csv`. Aggregates and statistical tests are written to `results/metrics_summary.csv` and `results/statistical_tests.csv`.

## Reproducibility Notes

The runner sets Python, NumPy, and PyTorch seeds for each training/evaluation condition. CuDNN deterministic mode is enabled. Environment metadata, package versions, CUDA visibility, GPU memory, and all experiment hyperparameters are saved in `results/environment.json`.

The default run trains two model variants across two datasets and three seeds. Use `--quick` for a small smoke test, but note that it overwrites the result files.
