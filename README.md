# PA-AHN: Prior-Anchored Adaptive Hybrid Normalization

This repository contains the implementation of **Prior-Anchored Adaptive Hybrid Normalization (PA-AHN)** for convolutional neural networks. PA-AHN performs input-conditioned fusion of Batch Normalization (BN), Layer Normalization (LN), and Instance Normalization (IN) using safe-prior logits, a residual controller, temperature-scaled softmax weighting, and entropy-based stabilization.

## Overview

PA-AHN extends Switchable Normalization-style BN/LN/IN hybrid normalization by making the normalization weights input-dependent. Instead of learning only static layer-wise mixture weights, PA-AHN uses Global Average Pooling (GAP) features and a residual MLP controller to generate adaptive residual logits. These residual logits are combined with learnable safe-prior base logits to improve stability during training.

The method is evaluated against **Switchable Normalization (SN)** on CIFAR-10 and CIFAR-100 using ResNet18 and ResNet50 across multiple batch sizes.

## Architecture

The PA-AHN layer contains:

* Parallel BN, LN, and IN branches
* GAP-based controller input
* Residual MLP controller
* Learnable safe-prior base logits
* Learnable residual scale
* Temperature-scaled softmax weighting
* Per-sample adaptive fusion of BN/LN/IN outputs
* Entropy-based L-Stab stabilization

The final safe-prior weights used in the experiments are:

```text
BN = 0.34, LN = 0.33, IN = 0.33
```

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* Python 3.x
* PyTorch
* torchvision
* numpy
* pandas
* matplotlib

## Datasets

The experiments use:

* CIFAR-10
* CIFAR-100

The datasets are automatically downloaded through `torchvision` when the training script is executed.

## Training PA-AHN

Example command:

```bash
python pa_ahn_normalization_toggle.py
```

The configuration inside the script can be modified to select:

* dataset: `cifar10` or `cifar100`
* model: `resnet18` or `resnet50`
* batch size: `8`, `32`, `64`, `128`, or `256`
* temperature
* L-Stab lambda
* number of epochs

## Training SN Baseline

Example command:

```bash
python sn_normalization.py
```

The SN baseline is used as the closest comparison method because it also combines BN, LN, and IN. Unlike PA-AHN, SN learns static layer-wise mixture weights rather than input-conditioned weights.

## Results

The main experiments compare PA-AHN and SN using:

* best test accuracy
* final test accuracy
* test loss
* entropy
* BN/LN/IN weight distribution
* parameter count
* average epoch time
* total training time
* average batch time
* peak GPU memory

Detailed results are reported in the associated research paper.

## Code Availability

GitHub repository:

```text
[Add GitHub link here]
```

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{PA_AHN_2026,
  title={Prior-Anchored Adaptive Hybrid Normalization for Convolutional Neural Networks},
  author={Wajid Mumtaz and Salman Abdul Ghafoor and Sajjad Hussain and Farhan Aftab},
  year={2026},
  note={Manuscript under preparation}
}
```

## License

This project is released under the MIT License.
