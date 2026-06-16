# Sampling Neural Network

This workspace tests whether sampling from intermediate neural activations changes model behavior compared with deterministic inference, final-softmax sampling, SAP-style activation sampling, Gaussian activation noise, and MC dropout.

## Key Findings

- Hidden activation sampling is not equivalent to final-softmax sampling.
- Naive categorical hidden sampling was highly disruptive: at `S=20`, accuracy fell from 98.55% to 79.41% on MNIST and from 88.77% to 66.30% on Fashion-MNIST.
- SAP-style and Gaussian hidden sampling preserved accuracy almost exactly, but increased entropy and stochastic probability variance.
- Final-softmax sampling preserved accuracy at `S=20`, but empirical sampled probabilities had worse NLL.
- The most useful next step is sample-aware training with the hidden sampling mechanism active during training.

See [REPORT.md](REPORT.md) for the full methodology, tables, statistical tests, figures, and limitations.

## Reproduce

Use the isolated environment in this workspace:

```bash
source .venv/bin/activate
python src/sampling_nn_experiment.py
```

For a quick smoke test:

```bash
source .venv/bin/activate
python src/sampling_nn_experiment.py --quick
```

The main run uses local datasets under `datasets/`, trains MNIST and Fashion-MNIST models for seeds `42`, `123`, and `456`, and writes outputs to `results/` and `figures/`.

## File Structure

- `planning.md`: preregistered motivation, novelty, and experiment plan.
- `src/sampling_nn_experiment.py`: training, inference interventions, metrics, statistics, and plots.
- `results/metrics_raw.csv`: seed-level experimental results.
- `results/metrics_summary.csv`: aggregate means, standard deviations, and confidence intervals.
- `results/statistical_tests.csv`: paired tests, effect sizes, and Holm-corrected p-values.
- `figures/`: generated plots used in the report.
- `REPORT.md`: primary research report.
