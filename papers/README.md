# Downloaded Papers

This directory contains PDFs collected for the "Sampling Neural Network" resource-finding phase. The most relevant papers are those that sample, perturb, prune, or model uncertainty in intermediate activations.

## Core Activation-Sampling and Activation-Uncertainty Papers

1. **Improving deep neural network performance through sampling**  
   Authors: Lakshmi A. Ghantasala, Ming-Che Li, Risi Jaiswal, Behtash Behin-Aein, Joseph Makin, Shreyas Sen, Supriyo Datta. Year: 2026.  
   File: `2026_ghantasala_sampling_dnn_performance.pdf`  
   Source: https://www.nature.com/articles/s44335-026-00063-7  
   Relevance: Directly replaces internal DNN activations with stochastic p-bit activations and studies multi-sample inference and sample-aware training on CIFAR-10 and generative models.

2. **Simple and Effective Stochastic Neural Networks**  
   Authors: Tianyuan Yu, Yongxin Yang, Da Li, Timothy Hospedales, Tao Xiang. Year: 2021.  
   File: `2021_yu_simple_effective_stochastic_neural_networks.pdf`  
   Source: https://ojs.aaai.org/index.php/AAAI/article/view/16436  
   Relevance: Learns Gaussian mean and variance for hidden activations, samples during forward passes, and evaluates pruning, robustness, label noise, and calibration.

3. **Activation-level uncertainty in deep neural networks**  
   Authors: Pablo Morales-Alvarez, Daniel Hernandez-Lobato, Rafael Molina, Jose Miguel Hernandez-Lobato. Year: 2021.  
   File: `2021_morales_activation_level_uncertainty.pdf`  
   Source: https://openreview.net/forum?id=UvBPbpvHRj-  
   Relevance: Moves uncertainty from weights to activation functions using 1D Gaussian processes.

4. **Stochastic Activation Pruning for Robust Adversarial Defense**  
   Authors: Guneet S. Dhillon, Kamyar Azizzadenesheli, Zachary C. Lipton, Jeremy Bernstein, Jean Kossaifi, Aran Khanna, Anima Anandkumar. Year: 2018.  
   File: `2018_dhillon_stochastic_activation_pruning.pdf`  
   Source: https://arxiv.org/abs/1803.01442  
   Relevance: Converts each activation map into a categorical distribution, samples surviving activations, and rescales them to preserve expectation.

5. **Noisy Activation Functions**  
   Authors: Caglar Gulcehre, Marcin Moczulski, Misha Denil, Yoshua Bengio. Year: 2016.  
   File: `2016_gulcehre_noisy_activation_functions.pdf`  
   Source: https://proceedings.mlr.press/v48/gulcehre16.html  
   Relevance: Adds activation-dependent noise to hidden nonlinearities, especially saturated units.

6. **How Sampling Impacts the Robustness of Stochastic Neural Networks**  
   Authors: Sina Daubener, Asja Fischer. Year: 2022.  
   File: `2022_daubener_sampling_robustness_snn.pdf`  
   Source: https://arxiv.org/abs/2204.10839  
   Relevance: Analyzes how sample sizes during attack and inference affect robustness of stochastic classifiers.

## Stochastic-Neuron and Discrete-Sampling Foundations

7. **Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation**  
   Authors: Yoshua Bengio, Nicholas Leonard, Aaron Courville. Year: 2013.  
   File: `2013_bengio_stochastic_neurons_conditional_computation.pdf`  
   Source: https://arxiv.org/abs/1308.3432  
   Relevance: Foundational estimators for stochastic or hard neurons, including straight-through and noise-injection approaches.

8. **Techniques for Learning Binary Stochastic Feedforward Neural Networks**  
   Authors: Tapani Raiko, Mathias Berglund, Guillaume Alain, Laurent Dinh. Year: 2015.  
   File: `2014_raiko_binary_stochastic_feedforward_networks.pdf`  
   Source: https://arxiv.org/abs/1406.2989  
   Relevance: Studies binary stochastic hidden units and the need for multiple hidden-activation samples.

9. **Categorical Reparameterization with Gumbel-Softmax**  
   Authors: Eric Jang, Shixiang Gu, Ben Poole. Year: 2017.  
   File: `2017_jang_gumbel_softmax.pdf`  
   Source: https://arxiv.org/abs/1611.01144  
   Relevance: Differentiable sampling from categorical distributions, useful if intermediate activations are normalized and sampled categorically.

10. **The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables**  
    Authors: Chris J. Maddison, Andriy Mnih, Yee Whye Teh. Year: 2017.  
    File: `2017_maddison_concrete_distribution.pdf`  
    Source: https://arxiv.org/abs/1611.00712  
    Relevance: Continuous relaxation for discrete stochastic nodes in computation graphs.

11. **Stochastic Neural Networks with Monotonic Activation Functions**  
    Authors: Siamak Ravanbakhsh, Barnabas Poczos, Jeff Schneider, Dale Schuurmans, Russell Greiner. Year: 2016.  
    File: `2016_ravanbakhsh_stochastic_monotonic_activations.pdf`  
    Source: https://arxiv.org/abs/1601.00034  
    Relevance: Builds stochastic units from smooth monotonic activations using Gaussian approximations.

## Dropout, Information Bottleneck, and Structural Baselines

12. **Variational Dropout and the Local Reparameterization Trick**  
    Authors: Diederik P. Kingma, Tim Salimans, Max Welling. Year: 2015.  
    File: `2015_kingma_variational_dropout_local_reparam.pdf`  
    Source: https://arxiv.org/abs/1506.02557  
    Relevance: Samples local pre-activation noise instead of global weights; a core stochastic baseline.

13. **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning**  
    Authors: Yarin Gal, Zoubin Ghahramani. Year: 2016.  
    File: `2016_gal_dropout_bayesian_approximation.pdf`  
    Source: https://arxiv.org/abs/1506.02142  
    Relevance: MC dropout is the standard test-time stochastic baseline.

14. **Information Dropout: Learning Optimal Representations Through Noisy Computation**  
    Authors: Alessandro Achille, Stefano Soatto. Year: 2016.  
    File: `2016_achille_information_dropout.pdf`  
    Source: https://arxiv.org/abs/1611.01353  
    Relevance: Uses multiplicative activation noise motivated by information-theoretic representation learning.

15. **Deep Variational Information Bottleneck**  
    Authors: Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy. Year: 2017.  
    File: `2017_alemi_deep_variational_information_bottleneck.pdf`  
    Source: https://arxiv.org/abs/1612.00410  
    Relevance: Treats an intermediate layer as a stochastic encoding and uses the reparameterization trick.

16. **Compressing Neural Networks using the Variational Information Bottleneck**  
    Authors: Bin Dai, Chen Zhu, David Wipf. Year: 2018.  
    File: `2018_dai_compressing_networks_vib.pdf`  
    Source: https://arxiv.org/abs/1802.10399  
    Relevance: Activation-level VIB pruning baseline used by SE-SNN.

17. **Deep Networks with Stochastic Depth**  
    Authors: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Q. Weinberger. Year: 2016.  
    File: `2016_huang_stochastic_depth.pdf`  
    Source: https://arxiv.org/abs/1603.09382  
    Relevance: Structural stochasticity baseline that randomly bypasses residual layers.

## Chunked Papers

The closest papers were chunked into `papers/pages/` using the provided PDF chunker with 3 pages per chunk:

- `2026_ghantasala_sampling_dnn_performance_*`
- `2021_yu_simple_effective_stochastic_neural_networks_*`
- `2021_morales_activation_level_uncertainty_*`
- `2018_dhillon_stochastic_activation_pruning_*`
- `2016_gulcehre_noisy_activation_functions_*`
- `2022_daubener_sampling_robustness_snn_*`
