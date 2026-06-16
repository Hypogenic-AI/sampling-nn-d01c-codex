# Downloaded Datasets

This directory contains local HuggingFace `datasets` snapshots for experiments. Data files are excluded from git by `datasets/.gitignore`; documentation and small sample JSON files are kept.

## Dataset 1: MNIST

### Overview

- Source: `ylecun/mnist` on HuggingFace
- Location: `datasets/mnist/`
- Size: train 60,000; test 10,000
- Format: HuggingFace Dataset saved with `save_to_disk`
- Task: 10-class image classification
- Why useful: Fast sanity-check benchmark for stochastic intermediate activation experiments.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("ylecun/mnist")
dataset.save_to_disk("datasets/mnist")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mnist")
```

### Sample Data

See `datasets/mnist/samples/examples.json`.

## Dataset 2: Fashion-MNIST

### Overview

- Source: `zalando-datasets/fashion_mnist` on HuggingFace
- Location: `datasets/fashion_mnist/`
- Size: train 60,000; test 10,000
- Format: HuggingFace Dataset saved with `save_to_disk`
- Task: 10-class image classification
- Why useful: Drop-in MNIST-scale test with harder visual classes; used in stochastic robustness literature.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("zalando-datasets/fashion_mnist")
dataset.save_to_disk("datasets/fashion_mnist")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/fashion_mnist")
```

### Sample Data

See `datasets/fashion_mnist/samples/examples.json`.

## Dataset 3: CIFAR-10

### Overview

- Source: `uoft-cs/cifar10` on HuggingFace
- Location: `datasets/cifar10/`
- Size: train 50,000; test 10,000
- Format: HuggingFace Dataset saved with `save_to_disk`
- Task: 10-class image classification
- Why useful: Primary benchmark in SAP, stochastic depth, p-DNN sampling, and many dropout baselines.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar10")
dataset.save_to_disk("datasets/cifar10")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cifar10")
```

### Sample Data

See `datasets/cifar10/samples/examples.json`.

## Dataset 4: CIFAR-100

### Overview

- Source: `uoft-cs/cifar100` on HuggingFace
- Location: `datasets/cifar100/`
- Size: train 50,000; test 10,000
- Format: HuggingFace Dataset saved with `save_to_disk`
- Task: 100-class image classification, with coarse labels also available
- Why useful: Calibration and compression benchmark used in SE-SNN and stochastic-depth work.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar100")
dataset.save_to_disk("datasets/cifar100")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cifar100")
```

### Sample Data

See `datasets/cifar100/samples/examples.json`.

## Validation Notes

All four datasets were loaded back from disk successfully. A machine-readable summary is in `datasets/dataset_summary.json`.
