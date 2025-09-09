## 项目简介

本项目提供用于模拟标签噪声（对称与非对称、多分类与二分类）的实用函数，方便在实验中构造含噪标签或基于给定噪声转移矩阵生成新标签。

文件：`asymmetric_noise.py`

## 功能概览

- 非对称二分类标签翻转：`noisify(y, p_minus, p_plus=None, random_state=0)`
- 基于转移矩阵的多分类翻转：`multiclass_noisify(y, P, random_state=0)`
- 构造统一（均匀）噪声矩阵：`build_uniform_P(size, noise)`
- CIFAR-100 相邻类转换矩阵：`build_for_cifar100(size, noise)`
- 行归一化工具：`row_normalize_P(P, copy=True)`
- 统一噪声注入（返回实际噪声率与矩阵）：`noisify_with_P(y_train, nb_classes, noise, random_state=None)`
- 数据集特定的非对称噪声：
  - MNIST：`noisify_mnist_asymmetric(y_train, noise, random_state=None)`
  - CIFAR-10：`noisify_cifar10_asymmetric(y_train, noise, random_state=None)`
  - CIFAR-100（同超类内转换）：`noisify_cifar100_asymmetric(y_train, noise, random_state=None)`
- 二分类固定非对称噪声：`noisify_binary_asymmetric(y_train, noise, random_state=None)`

## 环境依赖

- Python 3.7+
- numpy

安装：

```bash
pip install numpy
```

## 快速上手

```python
import numpy as np
from asymmetric_noise import (
    noisify, build_uniform_P, multiclass_noisify,
    noisify_with_P, noisify_mnist_asymmetric,
    noisify_cifar10_asymmetric, noisify_cifar100_asymmetric,
    noisify_binary_asymmetric
)

# 1) 二分类非对称噪声示例（标签取值必须为 -1 / +1）
y_binary = np.random.choice([-1, 1], size=1000)
y_binary_noisy = noisify(y_binary, p_minus=0.2, p_plus=0.05, random_state=42)

# 2) 多分类统一噪声（均匀翻转）
num_classes = 10
y_multi = np.random.randint(0, num_classes, size=1000)
P = build_uniform_P(num_classes, noise=0.4)  # 每一类以均匀方式被翻转
y_multi_noisy = multiclass_noisify(y_multi, P=P, random_state=42)

# 3) 自动生成统一噪声并返回实际噪声率与 P
y_multi_noisy2, P2 = noisify_with_P(y_multi, nb_classes=num_classes, noise=0.3, random_state=42)

# 4) 数据集特定的非对称噪声（示例）
y_mnist = np.random.randint(0, 10, size=1000)
y_mnist_noisy, P_mnist = noisify_mnist_asymmetric(y_mnist, noise=0.3, random_state=42)

y_cifar10 = np.random.randint(0, 10, size=1000)
y_cifar10_noisy, P_cifar10 = noisify_cifar10_asymmetric(y_cifar10, noise=0.2, random_state=42)

y_cifar100 = np.random.randint(0, 100, size=1000)
y_cifar100_noisy, P_cifar100 = noisify_cifar100_asymmetric(y_cifar100, noise=0.1, random_state=42)

# 5) 二分类固定非对称噪声（1->0 概率为 n，0->1 固定为 0.05）
y_bin01 = np.random.randint(0, 2, size=1000)
y_bin01_noisy, P_bin = noisify_binary_asymmetric(y_bin01, noise=0.2, random_state=42)
```

## 重要说明

- `noisify` 要求二分类标签为 {-1, +1}；其他多分类相关函数则要求标签为 [0, K-1] 的整数。
- 传入的转移矩阵 `P` 必须为行随机矩阵（每行元素非负、行和为 1）。
- 多数函数会打印“实际噪声率”（新旧标签不等的比例），便于核对设定噪声与实际效果。

## 许可

若无特别说明，默认为 MIT 许可。可按需修改。

## 致谢

本文件改编与整理自常见的标签噪声注入实现，便于研究复现与教学示例。


