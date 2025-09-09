## CPSL

This repository is an official implementation of the paper: [CPSL: A semi-supervised framework with class prototype-based modeling for combating noisy labels](https://www.sciencedirect.com/science/article/abs/pii/S0031320325009793).

If you find this repository useful in your research, please cite:

```
Xia Q, Lee F, Xie L, et al. CPSL: A Semi-supervised Framework with Class Prototype-based Modeling for Combating Noisy Labels[J]. Pattern Recognition, 2025: 112318.
```

The effectiveness of deep learning models is primarily attributed to the development of datasets free from disturbances. However, these datasets contain noisy annotations due to mistakes in manual labeling. We propose a novel learning framework for learning with noisy labels in image classification tasks, called Class Prototype-based Modeling (CPM) for Semi-supervised Learning (CPSL). Experimental results on corrupted datasets, like CIFAR-10, CIFAR-100, Tiny-ImageNet, Animal10N and Clothing1M, demonstrate the effectiveness of our approach, particularly in challenging datasets, such as too-large categories and real-world scenarios.

![overview](docs/CPSL.jpg)

### Setups

The environment is as bellow:

- Python
- Pytorch

### Running CPSL on benchmark datasets

Here is an example:

```bash
python - <<'PY'
import numpy as np
from asymmetric_noise import (
    build_uniform_P, multiclass_noisify,
    noisify_mnist_asymmetric, noisify_cifar10_asymmetric,
    noisify_cifar100_asymmetric, noisify, noisify_binary_asymmetric
)

# Multi-class with uniform noise
K = 10
y = np.random.randint(0, K, size=1000)
P = build_uniform_P(K, noise=0.4)
y_noisy = multiclass_noisify(y, P, random_state=42)
print('Noisy labels (head):', y_noisy[:10])

# Binary asymmetric noise with labels in {-1, +1}
y_bin = np.random.choice([-1, 1], size=1000)
y_bin_noisy = noisify(y_bin, p_minus=0.2, p_plus=0.05, random_state=42)
print('Binary noisy labels (head):', y_bin_noisy[:10])
PY
```

### Reference

[1]: Thank for TCC-net: A two-stage training method with contradictory loss and co-teaching based on meta-learning for learning with noisy labels.
