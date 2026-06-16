# Resources Catalog

## Summary

This document catalogs resources gathered for the "Sampling Neural Network" research project.

- Papers downloaded: 17
- Datasets downloaded: 4
- Code repositories cloned: 7
- Local environment: `.venv` created with `uv`; dependencies tracked in `pyproject.toml`

## Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Improving deep neural network performance through sampling | Ghantasala et al. | 2026 | `papers/2026_ghantasala_sampling_dnn_performance.pdf` | Direct p-bit activation sampling and sample-aware training. |
| Simple and Effective Stochastic Neural Networks | Yu et al. | 2021 | `papers/2021_yu_simple_effective_stochastic_neural_networks.pdf` | Gaussian activation uncertainty, robustness, pruning, calibration. |
| Activation-level uncertainty in deep neural networks | Morales-Alvarez et al. | 2021 | `papers/2021_morales_activation_level_uncertainty.pdf` | Deterministic weights with stochastic activation functions. |
| How Sampling Impacts the Robustness of Stochastic Neural Networks | Daubener, Fischer | 2022 | `papers/2022_daubener_sampling_robustness_snn.pdf` | Evaluation theory for stochastic classifiers and sample sizes. |
| Stochastic Activation Pruning for Robust Adversarial Defense | Dhillon et al. | 2018 | `papers/2018_dhillon_stochastic_activation_pruning.pdf` | Categorical sampling over activation magnitudes. |
| Noisy Activation Functions | Gulcehre et al. | 2016 | `papers/2016_gulcehre_noisy_activation_functions.pdf` | Activation-dependent noise and annealing. |
| Variational Dropout and the Local Reparameterization Trick | Kingma et al. | 2015 | `papers/2015_kingma_variational_dropout_local_reparam.pdf` | Local pre-activation sampling baseline. |
| Dropout as a Bayesian Approximation | Gal, Ghahramani | 2016 | `papers/2016_gal_dropout_bayesian_approximation.pdf` | MC dropout uncertainty baseline. |
| Information Dropout | Achille, Soatto | 2016 | `papers/2016_achille_information_dropout.pdf` | Multiplicative activation-noise representation learning. |
| Deep Variational Information Bottleneck | Alemi et al. | 2017 | `papers/2017_alemi_deep_variational_information_bottleneck.pdf` | Stochastic intermediate encoding baseline. |
| Compressing Neural Networks using the Variational Information Bottleneck | Dai et al. | 2018 | `papers/2018_dai_compressing_networks_vib.pdf` | VIB activation pruning baseline. |
| Categorical Reparameterization with Gumbel-Softmax | Jang et al. | 2017 | `papers/2017_jang_gumbel_softmax.pdf` | Differentiable categorical sampling. |
| The Concrete Distribution | Maddison et al. | 2017 | `papers/2017_maddison_concrete_distribution.pdf` | Differentiable discrete relaxation. |
| Estimating or Propagating Gradients Through Stochastic Neurons | Bengio et al. | 2013 | `papers/2013_bengio_stochastic_neurons_conditional_computation.pdf` | Stochastic-neuron gradient estimators. |
| Techniques for Learning Binary Stochastic Feedforward Neural Networks | Raiko et al. | 2015 | `papers/2014_raiko_binary_stochastic_feedforward_networks.pdf` | Binary stochastic hidden-unit training. |
| Stochastic Neural Networks with Monotonic Activation Functions | Ravanbakhsh et al. | 2016 | `papers/2016_ravanbakhsh_stochastic_monotonic_activations.pdf` | Stochastic monotonic activation units. |
| Deep Networks with Stochastic Depth | Huang et al. | 2016 | `papers/2016_huang_stochastic_depth.pdf` | Layer/block stochasticity baseline. |

See `papers/README.md` for descriptions and source URLs.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MNIST | HuggingFace `ylecun/mnist` | 60k train, 10k test | 10-class image classification | `datasets/mnist/` | Fast sanity-check benchmark. |
| Fashion-MNIST | HuggingFace `zalando-datasets/fashion_mnist` | 60k train, 10k test | 10-class image classification | `datasets/fashion_mnist/` | Lightweight robustness benchmark. |
| CIFAR-10 | HuggingFace `uoft-cs/cifar10` | 50k train, 10k test | 10-class image classification | `datasets/cifar10/` | Primary benchmark for SAP and p-DNN sampling. |
| CIFAR-100 | HuggingFace `uoft-cs/cifar100` | 50k train, 10k test | 100-class image classification | `datasets/cifar100/` | Calibration and harder classification benchmark. |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| stochastic-activation-pruning | https://github.com/Guneet-Dhillon/Stochastic-Activation-Pruning | Official SAP notebook | `code/stochastic-activation-pruning/` | MXNet notebook; useful formula reference. |
| noisy-units | https://github.com/caglar/noisy_units | Official noisy activation code | `code/noisy-units/` | TensorFlow/Theano implementations. |
| auNN | https://github.com/pablomorales92/auNN | Official activation-level uncertainty code | `code/aunn/` | TensorFlow 1.x/Python 3.6. |
| vib-demo | https://github.com/alexalemi/vib_demo | VIB demo notebooks | `code/vib-demo/` | TensorFlow and JAX notebooks. |
| stochastic-depth | https://github.com/yueatsprograms/stochastic_depth | Official stochastic depth code | `code/stochastic-depth/` | Torch 7 implementation. |
| bayesian-neural-networks | https://github.com/JavierAntoran/Bayesian-Neural-Networks | Bayesian and MC dropout baselines | `code/bayesian-neural-networks/` | Python 2.7/PyTorch 1.0.1; useful reference. |
| pytorch-cifar | https://github.com/kuangliu/pytorch-cifar | CIFAR baseline models | `code/pytorch-cifar/` | Practical starting point for modern PyTorch adaptation. |

See `code/README.md` for detailed notes.

## Search Strategy

The requested paper-finder script was attempted first in diligent JSON mode, but it stalled without output and was terminated. Manual search then covered arXiv, OpenReview, PMLR, NeurIPS proceedings, AAAI, Nature, Papers with Code style searches, and GitHub.

Search phrases included:

- "stochastic activation neural network"
- "sampling hidden layer activations neural networks"
- "stochastic activation pruning"
- "activation-level uncertainty"
- "noisy activation functions"
- "local reparameterization trick variational dropout"
- "Gumbel-Softmax categorical reparameterization"
- "stochastic neurons conditional computation"

## Selection Criteria

- Direct relevance to stochastic intermediate activations.
- Utility as an experimental baseline.
- Accessible PDF and reproducible benchmark.
- Code availability when possible.
- Preference for lightweight datasets that a downstream automated runner can train on.

## Challenges Encountered

- The paper-finder service and the arXiv Python client both stalled in this environment, so web and direct source searches were used.
- Several official code repositories are from older ecosystems: TensorFlow 1.x, Theano, MXNet, Torch 7, or Python 2.7.
- Running original code was not attempted because setting up multiple obsolete frameworks would pollute the fresh resource environment and is not necessary for this phase.

## Recommendations for Experiment Design

1. Primary datasets: MNIST for smoke tests, CIFAR-10 for the main result, CIFAR-100 for calibration follow-up.
2. Baselines: deterministic, final-softmax sampling, MC dropout, Gaussian activation noise, SAP-style sampling, VIB/Information Dropout, local reparameterization.
3. Metrics: accuracy, NLL, ECE, predictive entropy, sample variance, runtime/sample count, robust accuracy under expectation-over-samples attacks.
4. Code to reuse: implement in modern PyTorch using `code/pytorch-cifar` or a small fresh model; use SAP/noisy-units/auNN repos for formulas, not direct execution.
5. Key experimental variable: location and form of the intermediate activation distribution.
