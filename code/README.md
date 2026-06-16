# Cloned Repositories

No code repositories were explicitly supplied in the research specification, so these were selected from official paper pages, GitHub search, and baseline needs.

## stochastic-activation-pruning

- URL: https://github.com/Guneet-Dhillon/Stochastic-Activation-Pruning
- Location: `code/stochastic-activation-pruning/`
- Purpose: Official SAP implementation notebook for "Stochastic Activation Pruning for Robust Adversarial Defense."
- Key files: `SAP.ipynb`
- Notes: Uses MXNet, CIFAR-10 record files, and GPU contexts in the notebook. Useful for the exact SAP sampling rule even if it needs porting to modern PyTorch.

## noisy-units

- URL: https://github.com/caglar/noisy_units
- Location: `code/noisy-units/`
- Purpose: Official code for "Noisy Activation Functions."
- Key files: `codes/tf/nunits.py`, `codes/theano/nunits.py`
- Notes: Provides TensorFlow and Theano versions of noisy hard-tanh and hard-sigmoid units. The code targets older TensorFlow/Theano APIs, but the formulas are easy to port.

## auNN

- URL: https://github.com/pablomorales92/auNN
- Location: `code/aunn/`
- Purpose: Official code for activation-level uncertainty with stochastic activation functions.
- Key files: `run.py`, `code/auNN.py`, `train_data.npz`
- Notes: Requires Python 3.6 and TensorFlow 1.x according to README. Treat as reference implementation unless a compatibility environment is created.

## vib-demo

- URL: https://github.com/alexalemi/vib_demo
- Location: `code/vib-demo/`
- Purpose: Demo implementation for Deep Variational Information Bottleneck.
- Key files: `MNISTVIB.ipynb`, `JAX_DVIB_MNIST.ipynb`, `VIBDemo2021.ipynb`
- Notes: Notebook-based. Useful for VIB-style stochastic latent/intermediate representation training.

## stochastic-depth

- URL: https://github.com/yueatsprograms/stochastic_depth
- Location: `code/stochastic-depth/`
- Purpose: Official Torch 7 code for "Deep Networks with Stochastic Depth."
- Key files: `main.lua`, `ResidualDrop.lua`, `cifar-dataset.lua`
- Notes: Requires Torch 7, CUDA packages, and Torch-format datasets. This is mainly a structural stochasticity reference.

## bayesian-neural-networks

- URL: https://github.com/JavierAntoran/Bayesian-Neural-Networks
- Location: `code/bayesian-neural-networks/`
- Purpose: PyTorch baselines for Bayes by Backprop, local reparameterization, MC dropout, SGLD, SGHMC, Laplace, and ensembles.
- Key files: `train_MCDropout_MNIST.py`, `train_BayesByBackprop_MNIST.py`, `train_SGLD_MNIST.py`, `src/`
- Notes: README states Python 2.7 and PyTorch 1.0.1. The local reparameterization implementation is especially relevant because it samples activations rather than weights.

## pytorch-cifar

- URL: https://github.com/kuangliu/pytorch-cifar
- Location: `code/pytorch-cifar/`
- Purpose: Lightweight PyTorch CIFAR training baseline with common CNN architectures.
- Key files: `main.py`, `models/`, `utils.py`
- Notes: Good starting point for implementing new intermediate activation sampling modules on CIFAR-10.

## Feasibility Notes

The official direct-method repos are mostly old TensorFlow, Theano, MXNet, or Torch 7 code. For the experiment runner, the most pragmatic route is likely to implement the activation sampler in modern PyTorch using `code/pytorch-cifar` or a fresh small model, while using the official repos to verify formulas and baseline behavior.
