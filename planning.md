# Sampling Neural Network: Research Plan

## Motivation & Novelty Assessment

### Why This Research Matters

Neural networks are usually deterministic inside the model and stochastic only at the output when a class or token is sampled from a final softmax. If hidden activations can also be treated as a distribution to sample from, the model may express uncertainty, robustness, or diversity at a representational level rather than only at the final decision. This matters for reliable ML systems because activation-level stochasticity could change calibration, confidence, robustness, and sample efficiency without requiring fully Bayesian weights.

### Gap in Existing Work

The gathered literature shows several related mechanisms: stochastic activation pruning (SAP), Gaussian activation uncertainty, noisy activations, MC dropout, variational dropout, stochastic depth, and variational information bottlenecks. The gap is that these methods are rarely compared against final-softmax sampling as the isolated control, and many evaluate a specialized training objective rather than asking the direct user question: what happens if we sample from an intermediate activation distribution in an otherwise standard network?

### Our Novel Contribution

This execution tests hidden activation sampling as an intervention family on the same trained CNN architecture. The core contrast is between output-only stochasticity and hidden-layer stochasticity, including categorical hidden-unit sampling, SAP-style magnitude-proportional masking, Gaussian activation noise, and MC dropout. The contribution is not a claim that stochastic hidden units are new; it is a focused empirical isolation of the behavioral effect of sampling from intermediate activations compared with sampling only at the final softmax.

### Experiment Justification

- Experiment 1: Train deterministic and dropout CNNs on MNIST and Fashion-MNIST. This verifies the pipeline and creates controlled base models for hidden sampling.
- Experiment 2: Evaluate deterministic argmax, final-softmax sampling, categorical hidden sampling, SAP-style hidden sampling, Gaussian activation noise, and MC dropout. This directly tests whether hidden sampling changes accuracy, NLL, calibration, entropy, and prediction variability beyond output sampling.
- Experiment 3: Sweep Monte Carlo sample count `S in {1, 5, 20}` at inference. This tests whether hidden sampling behaves like final softmax sampling, where averaging more samples should stabilize predictions, or whether it introduces persistent representation noise.
- Experiment 4: Run each condition across three seeds and paired statistical tests. This guards against interpreting single-run stochastic artifacts as method effects.

## Research Question

Does sampling from a distribution over intermediate neural activations change classification behavior compared with deterministic inference, final-softmax sampling, dropout, and simple activation-noise baselines?

## Background and Motivation

Prior work establishes that stochastic internal representations can be useful for robustness, uncertainty, pruning, and calibration. SAP samples activation units with probability proportional to activation magnitude, Gaussian stochastic neural networks predict activation uncertainty, MC dropout keeps hidden stochasticity active at inference, and information bottlenecks inject noise into latent representations. The user's framing is narrower and experimentally useful: sample the intermediate activation distribution in a softmax-like way and compare the behavioral consequences directly to output softmax sampling.

## Hypothesis Decomposition

- H1: Hidden activation sampling changes predictive behavior relative to deterministic inference, measured by accuracy, NLL, ECE, predictive entropy, and probability variance.
- H2: Hidden activation sampling is not equivalent to final-softmax sampling; it changes internal representations before the classifier head and should therefore produce different calibration and error patterns.
- H3: Monte Carlo averaging reduces variance for hidden stochastic methods, but may not fully recover deterministic performance if the hidden sampling distribution discards important representational information.
- H4: SAP-style magnitude-proportional sampling should be less damaging than uniform or categorical one-hot-style hidden sampling because it preserves high-magnitude activations in expectation.

Independent variables are dataset, seed, model type, sampling method, sampling location, and Monte Carlo sample count. Dependent variables are accuracy, NLL, ECE, entropy, Brier score, probability variance, inference time, and per-class accuracy.

## Proposed Methodology

### Approach

Use a compact PyTorch CNN for MNIST-scale image classification so multiple seeds and stochastic inference sweeps can run in a single session. Train a deterministic CNN and a CNN with dropout on MNIST and Fashion-MNIST. At evaluation time, insert stochastic hidden sampling at the penultimate activation vector because this is the closest analogue to sampling over a final softmax class distribution while keeping the classifier head fixed.

### Experimental Steps

1. Validate environment, GPU, package versions, and dataset availability.
2. Load MNIST and Fashion-MNIST from local HuggingFace `save_to_disk` snapshots.
3. Create stratified train/validation splits from the original training set and keep the official test set for final evaluation.
4. Train deterministic and dropout CNNs for each seed with early stopping on validation loss.
5. Evaluate each saved model under deterministic argmax and output-softmax sampling.
6. Evaluate hidden interventions at the penultimate representation:
   - `hidden_categorical`: sample one active dimension from `softmax(abs(h) / tau)` and rescale by the inverse sampling probability.
   - `sap`: sample a Bernoulli mask over hidden dimensions with probabilities proportional to absolute activations and a target keep fraction, then rescale survivors.
   - `gaussian`: add zero-mean Gaussian activation noise proportional to hidden activation standard deviation.
   - `mc_dropout`: keep dropout active during inference for the dropout-trained model.
7. Repeat stochastic inference for sample counts `S = 1, 5, 20`, averaging probabilities across samples.
8. Analyze method differences across seeds with paired tests, effect sizes, confidence intervals, and Holm correction for multiple comparisons.

### Baselines

- Deterministic CNN with argmax output: main non-stochastic control.
- Deterministic CNN with final-softmax class sampling: isolates the common output sampling mechanism.
- Dropout CNN with dropout disabled at inference: standard regularized baseline.
- MC dropout at inference: standard hidden stochasticity baseline.
- Gaussian activation noise: simple continuous activation perturbation baseline.
- SAP-style activation sampling: closest prior work to sampling hidden activations as a distribution.

### Evaluation Metrics

- Accuracy: primary task performance.
- Negative log likelihood: probabilistic quality.
- Expected calibration error with 10 bins: confidence calibration.
- Brier score: probabilistic calibration and sharpness.
- Predictive entropy: uncertainty induced by stochastic methods.
- Mean probability variance across Monte Carlo samples: stochastic prediction diversity.
- Per-class accuracy and confusion matrices: error analysis.
- Runtime per test pass: computational cost of Monte Carlo inference.

### Statistical Analysis Plan

For each dataset and method, compute mean and standard deviation across three seeds. Compare each stochastic method to deterministic inference with paired tests on seed-level metrics. With only three paired runs, use paired t-tests as a descriptive parametric test and Wilcoxon signed-rank where SciPy supports it; interpret p-values cautiously and emphasize effect sizes and confidence intervals. Apply Holm correction within each dataset/metric family. Report Cohen's dz for paired differences.

### Resource Plan

Use the existing `.venv` and `uv` dependency tracking. Use GPU `cuda:0` with batch size 256 for MNIST-scale training on RTX A6000 hardware. Store code in `src/`, results in `results/`, figures in `figures/`, logs in `logs/`, and checkpoints in `results/checkpoints/`.

## Expected Outcomes

Results supporting the hypothesis would show hidden sampling shifts calibration, entropy, variance, or accuracy differently from final-softmax sampling. Results refuting the stronger version would show hidden sampling mostly harms performance without improving uncertainty or calibration. A likely outcome is mixed: SAP and Gaussian noise may preserve accuracy better than categorical hidden sampling, while all hidden stochastic methods increase entropy and sample variance relative to deterministic inference.

## Timeline and Milestones

- Planning and environment: completed first.
- Implementation: build dataset loader, model, sampling modules, training loop, evaluation harness, and analysis scripts.
- Experimentation: run two datasets, three seeds, and all inference conditions.
- Analysis: generate summary tables, figures, and statistical comparisons.
- Documentation: write final `REPORT.md`, `README.md`, and code walkthrough notes where useful.

## Potential Challenges

- Full CIFAR-10 training could exceed the session budget, so CIFAR is deferred unless MNIST/Fashion-MNIST completes quickly.
- Hidden categorical one-sample vectors may be too destructive; expectation-preserving rescaling and Monte Carlo averaging mitigate this but do not guarantee useful performance.
- Stochastic methods can appear robust or calibrated due to noise alone; NLL, ECE, entropy, and probability variance are reported together to avoid overclaiming.
- With three seeds, statistical power is limited; effect sizes and raw seed-level results will be emphasized over binary significance claims.

## Success Criteria

The session succeeds if it produces reproducible code, actual trained-model results on at least MNIST and Fashion-MNIST, saved metrics and figures, and a report that clearly answers whether intermediate activation sampling behaves differently from final-softmax sampling and other stochastic baselines.
