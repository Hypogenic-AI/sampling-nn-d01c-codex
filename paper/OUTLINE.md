# Sampling Neural Network Paper Outline

## Title
- Emphasize the main empirical finding: hidden activation sampling is not equivalent to output sampling.

## Abstract
- Context: stochastic predictions are usually discussed at the final softmax, but hidden activations can also be stochastic.
- Gap: prior hidden-stochastic methods are rarely compared directly to final-softmax sampling as an isolated control.
- Approach: compact CNNs on MNIST and Fashion-MNIST, deterministic and dropout variants, six inference conditions, three seeds, S in {1, 5, 20}.
- Results: categorical hidden sampling drops accuracy by 19.13 pp on MNIST and 22.46 pp on Fashion-MNIST; SAP and Gaussian sampling preserve accuracy; output sampling preserves accuracy but hurts empirical NLL.
- Significance: intermediate stochasticity is a representation-level intervention requiring careful distributions and training objectives.

## Introduction
- Hook: sampling from a softmax is common at the output, but moving sampling inside the network changes the computation.
- Importance: calibration, uncertainty, and robustness depend on where stochasticity enters.
- Gap: existing work studies SAP, Gaussian activation uncertainty, dropout, and stochastic bottlenecks, but seldom isolates hidden sampling against final-softmax sampling.
- Approach: evaluate hidden categorical, SAP, Gaussian, output sampling, dropout evaluation, and MC dropout in the same pipeline.
- Preview: hidden categorical strongly increases entropy/ECE and reduces accuracy; SAP/Gaussian preserve accuracy; output sampling differs by preserving decision behavior while degrading sampled NLL.
- Contributions:
  - We define a controlled comparison between output-only and hidden-layer sampling.
  - We evaluate multiple hidden distributions under matched CNNs and sample counts.
  - We report calibration, entropy, variance, NLL, and paired seed-level statistics.
  - We identify sample-aware training and structured sampling as the next required step.

## Related Work
- Activation sampling/pruning: SAP and p-bit sampling.
- Activation uncertainty/noise: SE-SNN, activation-level uncertainty, noisy activations, local reparameterization.
- Dropout and stochastic bottlenecks: MC dropout, Information Dropout, VIB, stochastic depth.
- Discrete stochastic nodes: Bengio estimators, Gumbel-Softmax, Concrete distribution.
- Positioning: this paper is not claiming hidden stochasticity is new; it isolates the output-sampling comparison.

## Methodology
- Formalize a classifier with penultimate vector h and classifier head.
- Describe deterministic inference and final-softmax sampling.
- Describe hidden categorical sampling from softmax(abs(h)/tau), SAP Bernoulli masking, Gaussian activation noise, dropout_eval, and MC dropout.
- Datasets: MNIST and Fashion-MNIST local snapshots, 20k train, 5k validation, 10k test, train-only normalization.
- Model/training: compact CNN, 128-dimensional penultimate vector, dropout p=0.25 variant, seeds 42/123/456, batch size 256, AMP.
- Metrics: accuracy, NLL, ECE, Brier, entropy, probability variance, runtime, paired tests with Holm correction.
- Evidence: dataset table, method overview figure.

## Results
- Primary table: S=20 for stochastic methods and deterministic/dropout baselines.
- Primary figures: accuracy and ECE plots.
- Sample-count table/figure: accuracy improves with S but categorical hidden remains far below deterministic.
- Statistical results: paired differences for accuracy, ECE, entropy, and NLL; p-values descriptive due to n=3.
- Interpretation within results: output sampling and hidden sampling diverge sharply.

## Discussion
- Interpret hidden categorical failure as representation destruction in a normally trained network.
- Explain why SAP/Gaussian are less damaging: they preserve more of h in expectation or perturb continuously.
- Discuss output sampling NLL caveat: empirical sampled probabilities are coarse count distributions, not analytic softmax probabilities.
- Limitations: two datasets, small CNN, penultimate-only sampling, three seeds, no robustness attacks, post-hoc sampling.
- Broader implications: stochastic hidden activations need training-aware objectives and attack-aware evaluation.

## Conclusion
- Restate controlled empirical answer: hidden sampling is behaviorally distinct from final-softmax sampling.
- Key takeaway: naive categorical hidden sampling is harmful, while SAP/Gaussian preserve accuracy and mostly alter uncertainty.
- Future work: sample-aware training, channel/spatial sampling, CIFAR-scale models, robustness with expectation-over-transformation attacks.

## Planned Tables and Figures
- Table 1: dataset splits and preprocessing.
- Figure 1: experimental pipeline and sampling locations.
- Table 2: primary metrics at S=20.
- Figure 2: primary accuracy and ECE.
- Table 3: accuracy versus sample count.
- Figure 3: entropy and probability variance sample sweeps.
- Table 4: key paired differences against deterministic inference.

