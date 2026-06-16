# Literature Review: Sampling Neural Network

## Review Scope

### Research Question

What is already known about sampling, perturbing, pruning, or modeling uncertainty in intermediate neural network activations, and how should experiments test whether sampling intermediate activations changes network behavior compared with deterministic inference, final-softmax sampling, dropout, and other stochastic baselines?

### Inclusion Criteria

- Methods that introduce stochasticity at hidden activations, stochastic neurons, or stochastic layers.
- Baselines that are standard for stochastic neural networks, uncertainty, robustness, or calibration.
- Papers with accessible PDFs, code, or reproducible datasets.
- Image classification papers using MNIST, Fashion-MNIST, CIFAR-10, or CIFAR-100.

### Exclusion Criteria

- Work only about output sampling without a hidden-layer mechanism.
- Hardware-only or theory-only work with no clear experimental transfer, unless it directly samples activations.
- Large-domain datasets that are impractical for a first experiment.

### Search Log

| Date | Query / Source | Results | Notes |
|------|----------------|---------|-------|
| 2026-06-16 | Paper-finder script: "sampling from intermediate layer activations stochastic neural networks noisy activations" | Stalled, no usable output | Fallback manual search used. |
| 2026-06-16 | Web/arXiv: stochastic activation, activation pruning, noisy activation functions | SAP, Noisy Activations, SE-SNN, activation-level uncertainty | Core literature. |
| 2026-06-16 | Web/arXiv: local reparameterization, dropout Bayesian approximation, information dropout | Variational dropout, MC dropout, Information Dropout | Baselines. |
| 2026-06-16 | Web/arXiv: Gumbel-Softmax, Concrete, stochastic neurons | Gumbel-Softmax, Concrete, Bengio stochastic neurons | Differentiable categorical sampling references. |
| 2026-06-16 | GitHub search | 7 repos cloned | Official and baseline implementations. |

## Research Area Overview

The hypothesis sits at the intersection of stochastic neural networks, activation uncertainty, dropout-style regularization, differentiable discrete sampling, adversarial robustness, and calibration. The most direct prior work already demonstrates that hidden activations can be treated as random variables. However, the precise proposed framing, "sample from an intermediate activation distribution in the same spirit as final softmax sampling," is still a useful experimental question because prior methods differ in how they define the distribution, whether sampling happens during training, whether samples are averaged at inference, and whether the goal is accuracy, robustness, uncertainty, compression, or hardware efficiency.

The strongest starting point is to treat the hypothesis as an intervention family rather than a single algorithm:

- **Categorical activation sampling**: normalize an activation vector or channel map into a distribution and sample indices or masks.
- **Gaussian activation sampling**: predict or estimate mean and variance for an intermediate representation and sample hidden states.
- **Multiplicative activation noise**: dropout, variational dropout, Information Dropout, or VIB-style stochastic latent encodings.
- **Structural stochasticity**: sample layers or blocks, as in stochastic depth.
- **Output-only stochasticity**: sample only from final softmax as a control.

## Key Papers

### Improving deep neural network performance through sampling

- Authors: Ghantasala et al.
- Year: 2026
- Key contribution: Introduces probabilistic DNNs using stochastic p-bit activations and sample-aware training.
- Methodology: Replaces standard activations with stochastic binary activations; averages multiple samples; analyzes energy tradeoffs.
- Datasets: CIFAR-10, CelebA, MNIST for hardware validation.
- Results: One 1-bit stochastic sample can match deterministic accuracy in their CIFAR-10 setting; multiple samples improve accuracy, with 2 samples reported as enough to outperform in the main classification setup.
- Relevance: Closest match to the proposed hypothesis because the stochasticity is in internal activations, not only weights or outputs.

### Simple and Effective Stochastic Neural Networks

- Authors: Yu et al.
- Year: 2021
- Key contribution: Models activation uncertainty by predicting Gaussian mean and variance at hidden layers and sampling during the forward pass.
- Methodology: Adds stochastic learning modules and regularizers that encourage useful activation variability.
- Datasets: MNIST, CIFAR-10, CIFAR-100.
- Baselines: VIBNet, variational dropout, Bayesian pruning, dropout calibration, adversarial defenses.
- Results: Competitive or improved pruning, adversarial defense, label-noise robustness, and calibration. Calibration stabilized after roughly 10 samples in the reported ResNet/CIFAR-100 setting.
- Relevance: Strong template for measuring more than accuracy: pruning, robustness, label noise, and expected calibration error.

### Activation-level uncertainty in deep neural networks

- Authors: Morales-Alvarez et al.
- Year: 2021
- Key contribution: Keeps weights deterministic and models activation functions with 1D Gaussian processes.
- Methodology: Defines auNNs where stochasticity is moved from weight space to activation functions.
- Datasets: UCI regression gap splits, large tabular datasets, classification examples.
- Baselines: Deterministic NNs, BNNs, functional BNNs, deep GPs.
- Results: Better uncertainty in gaps and out-of-distribution regions than weight-space baselines in their experiments.
- Relevance: Clear conceptual support for activation-level stochasticity as a distinct modeling axis.

### Stochastic Activation Pruning for Robust Adversarial Defense

- Authors: Dhillon et al.
- Year: 2018
- Key contribution: Defines SAP, a post-hoc stochastic activation sampling/pruning method.
- Methodology: For each activation map, sample activation indices with probability proportional to absolute activation magnitude, zero non-survivors, and rescale survivors by inverse sampling probability.
- Datasets: CIFAR-10 and reinforcement-learning tasks.
- Baselines: Dense model, dropout, adversarial training, weight pruning, random noisy activations, random scaled activations.
- Results: On CIFAR-10 ResNet-20, SAP gave absolute accuracy improvements under FGSM at small perturbation strengths, while sacrificing clean accuracy when sampling was too aggressive.
- Relevance: Directly operationalizes "sampling from an intermediate activation distribution."

### Noisy Activation Functions

- Authors: Gulcehre et al.
- Year: 2016
- Key contribution: Adds noise to activation functions, especially in saturated regions, to permit gradient flow and exploration.
- Methodology: Noisy hard-tanh and hard-sigmoid variants; optional annealing; deterministic expectation at test time.
- Datasets: MNIST, Penn Treebank, program execution, translation, captioning.
- Results: Improved optimization and task performance across several settings.
- Relevance: Strong baseline for activation noise, but less directly about sampling a probability distribution over activations.

### How Sampling Impacts the Robustness of Stochastic Neural Networks

- Authors: Daubener and Fischer
- Year: 2022
- Key contribution: Theoretical and empirical analysis of why stochastic classifiers can resist gradient-based attacks.
- Methodology: Defines stochastic classifiers as random functions whose predictions are Monte Carlo estimates; studies sample size during attack and inference.
- Datasets: Fashion-MNIST, CIFAR-10, CIFAR-100.
- Results: Robustness depends on margins, gradient norms, and angles between attack and inference decision boundaries; attack sample size matters more than inference sample size for evaluating stochastic defenses.
- Relevance: Important evaluation warning: stochastic activation methods must be attacked with expectation-over-transformation style gradients.

### Variational Dropout and the Local Reparameterization Trick

- Authors: Kingma, Salimans, Welling
- Year: 2015
- Key contribution: Converts global weight uncertainty into local activation noise, reducing stochastic-gradient variance.
- Methodology: Samples pre-activations independently per datapoint.
- Relevance: Essential baseline because it samples activations for Bayesian efficiency, even though the stochasticity originates from parameter uncertainty.

### Dropout as a Bayesian Approximation

- Authors: Gal and Ghahramani
- Year: 2016
- Key contribution: Interprets dropout as approximate Bayesian inference and enables MC dropout uncertainty at test time.
- Datasets: MNIST and regression/RL examples.
- Relevance: Standard stochastic baseline and control condition.

### Information Dropout and Deep VIB

- Authors: Achille and Soatto; Alemi et al.
- Years: 2016 and 2017
- Key contribution: Treats intermediate representations as noisy or compressed stochastic encodings.
- Methodology: Multiplicative activation noise or variational bottleneck using the reparameterization trick.
- Relevance: Strong baselines for "hidden stochastic representation" experiments.

### Gumbel-Softmax, Concrete, and stochastic-neuron estimators

- Authors: Jang et al.; Maddison et al.; Bengio et al.
- Years: 2013 and 2017
- Key contribution: Provide differentiable estimators for discrete stochastic nodes.
- Relevance: Useful if the proposed method samples one-hot or sparse categorical activation states during training rather than only at inference.

### Deep Networks with Stochastic Depth

- Authors: Huang et al.
- Year: 2016
- Key contribution: Randomly bypasses residual layers during training.
- Relevance: Structural stochastic baseline, especially for residual CNNs.

## Common Methodologies

- **Monte Carlo inference**: Draw S forward samples and average logits or probabilities.
- **Activation masking/pruning**: Sample hidden units or channels, zero the rest, and rescale survivors.
- **Gaussian latent activations**: Learn or estimate `mu` and `sigma`, sample `z = mu + sigma * epsilon`.
- **Categorical relaxation**: Use Gumbel-Softmax or Concrete for trainable categorical hidden choices.
- **Expectation-preserving rescaling**: Keep stochastic hidden activations unbiased in expectation when possible.
- **Sample-aware training**: Train with the same number or style of samples expected at inference.

## Standard Baselines

- Deterministic network with argmax output.
- Deterministic network with final-softmax class sampling only.
- Dropout trained normally and disabled at inference.
- MC dropout with dropout active at inference.
- Gaussian activation noise with fixed variance.
- SAP-style activation pruning.
- VIB or Information Dropout bottleneck.
- Local reparameterization variational dropout.
- Stochastic depth for residual networks.

## Evaluation Metrics

- Clean accuracy and negative log likelihood.
- Expected calibration error with 10 or 15 bins.
- Predictive entropy and mutual information for uncertainty.
- Sample variance of logits/probabilities across stochastic forward passes.
- Robust accuracy under FGSM/PGD with expectation-over-samples attacks.
- Accuracy versus number of samples, including latency/compute cost.
- For compression variants: retained parameters, FLOPs, activation memory.

## Datasets in the Literature

- MNIST: fast smoke test and compatibility with older stochastic-neuron baselines.
- Fashion-MNIST: useful for robustness analysis and still lightweight.
- CIFAR-10: primary image classification benchmark for SAP, p-DNN sampling, stochastic depth, and PyTorch baselines.
- CIFAR-100: useful for calibration and harder classification.

## Gaps and Opportunities

- Existing activation-sampling methods use different distributions and objectives; few compare categorical hidden activation sampling directly against final-softmax sampling as the isolated variable.
- Many papers train with hidden stochasticity, but fewer evaluate post-hoc hidden sampling on deterministic pretrained networks beyond SAP and p-DNN variants.
- Robustness claims can be misleading unless attacks use many stochastic samples.
- Calibration and sample variance are likely more informative than accuracy alone.
- The most relevant official code is old, so a modern PyTorch implementation will likely be cleaner than trying to run original environments.

## Recommendations for Experiments

- Start with MNIST and CIFAR-10 using a small MLP/CNN and a ResNet-style CIFAR model.
- Implement a small PyTorch `ActivationSampler` module with modes:
  - deterministic pass-through,
  - output softmax sampling only,
  - categorical activation sampling with Gumbel-Softmax training option,
  - SAP-style magnitude-proportional mask sampling,
  - Gaussian activation noise,
  - MC dropout.
- Test sampling locations separately: early hidden layer, penultimate representation, and all eligible layers.
- Sweep sample count `S in {1, 2, 5, 10, 25, 50, 100}` at inference.
- Report clean accuracy, NLL, ECE, predictive entropy, runtime, and robust accuracy under expectation-over-samples attacks.
- Compare post-hoc insertion into pretrained deterministic networks versus sample-aware training.
- Use CIFAR-100 only after CIFAR-10 behavior is stable, mainly for calibration.
