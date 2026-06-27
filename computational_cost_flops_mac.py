# ================================================================
# FILE: computational_cost_flops_mac.py
# Clean computational-cost comparison for PA-AHN, SN, BN, and GN
# on CIFAR-10 / CIFAR-100.
#
# Purpose:
# - Measure clean training-time computational cost without analysis/logging overhead.
# - No per-batch BN/LN/IN weight logging.
# - No CSV logs.
# - No plots.
# - No checkpoint saving.
# - No file writing except one final Excel summary table.
# - No .cpu().numpy() weight logging.
#
# What is printed:
# - Method name and configuration
# - Mathematical operations (FLOPs/MACs)
# - Extra parameters
# - Runtime overhead
# - Memory usage

#
# Important:
# - base_dir path is kept unchanged from thesis code.
# - This script assumes the following files already exist in the project:
#   models/resnet18_pa_ahn.py
#   models/resnet50_pa_ahn.py
#   norms/prior_anchored_adaptive_hybrid_normalization.py
#   models/resnet18_sn.py
#   models/resnet50_sn.py
#   norms/switchable_norm.py
#   models/resnet18_bn.py
#   models/resnet50_bn.py
#   models/resnet18_gn.py
#   models/resnet50_gn.py
# ================================================================

import os
import sys
import json
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------------------------------------
# Keep thesis project path unchanged
# ------------------------------------------------------------
BASE_DIR = "./runs"
sys.path.append(BASE_DIR)

# PA-AHN imports
from models.resnet18_pa_ahn import ResNet18_AHN
from models.resnet50_pa_ahn import ResNet50_AHN
from norms.prior_anchored_adaptive_hybrid_normalization import AdaptiveHybridNorm2d

# SN imports
from models.resnet18_sn import ResNet18_SN
from models.resnet50_sn import ResNet50_SN
from norms.switchable_norm import SwitchableNorm2d

# BN imports
from models.resnet18_bn import ResNet18_BN
from models.resnet50_bn import ResNet50_BN

# GN imports
from models.resnet18_gn import ResNet18_GN
from models.resnet50_gn import ResNet50_GN


# =============================
# Configuration
# =============================
@dataclass
class Config:
    base_dir: str = "/home/seecs/farhan/AHN_Thesis"

    dataset_name: str = "cifar10"       # options: cifar10
    num_classes: int = 10               # options: cifar10

    # dataset_name: str = "cifar100"      # options: cifar100
    # num_classes: int = 100              # options: cifar100

    model_type: str = "resnet18"        # options: "resnet18"
    # model_type: str = "resnet50"        # options: "resnet50"

    # batch_size: int = 8                 # options: batch_size 8
    # batch_size: int = 32                # options: batch_size 32
    # batch_size: int = 64                # options: batch_size 64
    # batch_size: int = 128               # options: batch_size 128
    batch_size: int = 256               # options: batch_size 256

    # num_epochs: int = 100               # option: epochs 100
    # num_epochs: int = 50                # option: epochs 50
    num_epochs: int = 20                # option: epochs 20

    num_workers: int = 2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42

    # PA-AHN settings
    # temperature: float = 1.2
    # temperature: float = 1.0
    temperature: float = 0.8
    # temperature: float = 0.7
    # temperature: float = 0.3

    # Safe prior weights used to initialize PA-AHN base logits
    safe_bn_weight: float = 0.34
    safe_ln_weight: float = 0.33
    safe_in_weight: float = 0.33

    # L-Stab setting for final PA-AHN training cost.
    # This is part of the PA-AHN training method, not just logging.
    use_lstab: bool = True
    lstab_lambda_max: float = 0.00042
    anneal_lambda: bool = False
    use_temperature_scaling: bool = True

    # GN setting: 32 is standard for ResNet channels 64/128/256/512.
    gn_groups: int = 32

    # SN paper-style batch-average inference calibration.
    # For clean TRAINING computational-cost comparison, default is False.
    # Set True only if you intentionally want to include SN calibration overhead.
    use_sn_batch_average_inference_calibration: bool = True
    sn_batch_average_calibration_batches: Optional[int] = None

    # Measurement settings
    measure_gpu_memory: bool = True

    # One CSV is updated after every completed method.
    output_csv_name: str = "gec_computational_cost_flops_macs_8configs_1epoch.csv"

    def __post_init__(self):
        self.num_classes = 10 if self.dataset_name.lower() == "cifar10" else 100
        self.lr = 0.1 * self.batch_size / 256


CFG = Config()

# Same paper-style linear learning-rate scaling used in PA-AHN, SN, BN and GN code.
# effective batch 256 -> lr 0.1
# effective batch 32  -> lr 0.0125
# effective batch 8   -> lr 0.003125
# cfg.lr is computed automatically in Config.__post_init__ for every run.


# =============================
# Utility functions
# =============================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(dataset_name: str):
    if dataset_name.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name.lower() == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform_train, transform_test


def get_dataloaders(cfg: Config):
    transform_train, transform_test = get_transforms(cfg.dataset_name)

    if cfg.dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif cfg.dataset_name.lower() == "cifar100":
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Test loader is intentionally created for configuration consistency,
    # but the clean cost script does not evaluate test accuracy/loss.
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    # Fresh one-time count of trainable parameters.
    # This is not taken from previous logs or CSV files.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


# =============================
# MACs / FLOPs estimation helpers
# =============================
# Convention used in this script:
# - MACs are multiply-accumulate operations.
# - FLOPs are commonly reported as 2 x MACs.
# - The CSV column is named "FLOPs / MACs" because committees often use these
#   terms loosely. Internally, the script stores MACs and also prints the convention.
# - "Forward Pass" is measured for ONE input image with shape [1, 3, 32, 32].
# - Training MACs per epoch are estimated as:
#       forward_macs_per_image x 3 x number_of_training_images
#   The factor 3 is a standard practical approximation for forward + backward.

TRAINING_BACKWARD_MULTIPLIER = 3.0


def _numel(shape) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return int(n)


def estimate_extra_ahn_macs(module: AdaptiveHybridNorm2d, inputs, output) -> int:
    """Extra PA-AHN operations not already counted by child BN/LN/IN/Linear hooks."""
    x = inputs[0]
    b, c, h, w = x.shape

    macs = 0
    # GAP controller input: approximate additions/divisions over HxW per channel.
    macs += b * c * h * w
    # Base logits + residual scale + softmax + fusion weights overhead.
    macs += b * 3 * 8
    # Weighted branch fusion: three branch multiplications + additions per element.
    macs += b * c * h * w * 5
    return int(macs)


def estimate_switchable_norm_macs(module: SwitchableNorm2d, inputs, output) -> int:
    """Approximate SN statistic-level mixing cost for one forward call."""
    x = inputs[0]
    b, c, h, w = x.shape
    elems = b * c * h * w

    macs = 0
    # IN mean and variance, LN/BN stat aggregation, statistic mixing, normalize, affine.
    macs += elems * 8
    macs += b * c * 6
    macs += b * 3 * 8
    return int(macs)


def estimate_forward_macs(model: nn.Module, device: torch.device, input_shape=(1, 3, 32, 32)) -> int:
    """
    Hook-based approximate MAC counter for this CIFAR ResNet codebase.
    It counts Conv2d, Linear, standard norm layers, and custom PA-AHN/SN overhead.
    """
    macs = {"total": 0}
    handles = []

    def conv_hook(module, inputs, output):
        x = inputs[0]
        batch_size = int(x.shape[0])
        out_channels = int(output.shape[1])
        out_h = int(output.shape[2])
        out_w = int(output.shape[3])
        kernel_h, kernel_w = module.kernel_size
        in_channels = int(module.in_channels)
        groups = int(module.groups)
        mac = batch_size * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w
        if module.bias is not None:
            mac += _numel(output.shape)
        macs["total"] += int(mac)

    def linear_hook(module, inputs, output):
        x = inputs[0]
        batch_size = int(x.shape[0]) if x.dim() > 1 else 1
        mac = batch_size * int(module.in_features) * int(module.out_features)
        if module.bias is not None:
            mac += batch_size * int(module.out_features)
        macs["total"] += int(mac)

    def norm_hook(module, inputs, output):
        # Approximate normalization cost: mean/var/normalize/affine.
        macs["total"] += int(_numel(output.shape) * 5)

    def ahn_extra_hook(module, inputs, output):
        macs["total"] += estimate_extra_ahn_macs(module, inputs, output)

    def sn_hook(module, inputs, output):
        macs["total"] += estimate_switchable_norm_macs(module, inputs, output)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            handles.append(m.register_forward_hook(norm_hook))
        elif isinstance(m, AdaptiveHybridNorm2d):
            handles.append(m.register_forward_hook(ahn_extra_hook))
        elif isinstance(m, SwitchableNorm2d):
            handles.append(m.register_forward_hook(sn_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        _ = model(dummy)
    if was_training:
        model.train()

    for h in handles:
        h.remove()

    return int(macs["total"])


def format_impact(value: float, baseline: float, unit: str = "") -> str:
    if baseline == 0:
        return "N/A"
    ratio = value / baseline
    pct = (ratio - 1.0) * 100.0
    suffix = f" {unit}" if unit else ""
    sign = "+" if pct >= 0 else ""
    return f"{ratio:.3f}x ({sign}{pct:.2f}% vs BN){suffix}"


def get_lambda_lstab(epoch: int, num_epochs: int, cfg: Config) -> float:
    if not cfg.use_lstab:
        return 0.0
    if cfg.anneal_lambda:
        return cfg.lstab_lambda_max * (1.0 - epoch / num_epochs)
    return cfg.lstab_lambda_max


def compute_ahn_entropy_loss(model: nn.Module) -> Tuple[torch.Tensor, int]:
    entropy_loss = None
    ahn_layers = 0

    for m in model.modules():
        if isinstance(m, AdaptiveHybridNorm2d):
            layer_entropy = m.compute_entropy()
            if entropy_loss is None:
                entropy_loss = layer_entropy
            else:
                entropy_loss = entropy_loss + layer_entropy
            ahn_layers += 1

    if entropy_loss is None:
        device = next(model.parameters()).device
        entropy_loss = torch.tensor(0.0, device=device)
    elif ahn_layers > 0:
        entropy_loss = entropy_loss / ahn_layers

    return entropy_loss, ahn_layers


def get_scaled_milestones(num_epochs: int) -> List[int]:
    ratios = [0.30, 0.60, 0.90]
    return sorted(set(max(1, int(round(num_epochs * r))) for r in ratios if int(round(num_epochs * r)) < num_epochs))


# ------------------------------------------------------------
# SN batch-average inference helpers kept for optional use.
# Default clean cost comparison does not call calibration.
# ------------------------------------------------------------
def start_batch_average_collection(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, SwitchableNorm2d):
            m.start_batch_average_collection()


def stop_batch_average_collection(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, SwitchableNorm2d):
            m.stop_batch_average_collection()


def finalize_batch_average(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, SwitchableNorm2d):
            m.finalize_batch_average()


def calibrate_sn_batch_average(model: nn.Module, train_loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> None:
    was_training = model.training
    model.train()
    start_batch_average_collection(model)

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    stop_batch_average_collection(model)
    finalize_batch_average(model)

    if not was_training:
        model.eval()


# =============================
# Model builders
# =============================
def build_model(method_name: str, cfg: Config, device: torch.device) -> nn.Module:
    method_name = method_name.upper()
    temperature = cfg.temperature if cfg.use_temperature_scaling else 1.0

    if method_name == "PA-AHN":
        if cfg.model_type == "resnet18":
            model = ResNet18_AHN(num_classes=cfg.num_classes, temperature=temperature).to(device)
        elif cfg.model_type == "resnet50":
            model = ResNet50_AHN(num_classes=cfg.num_classes, temperature=temperature).to(device)
        else:
            raise ValueError("Invalid model_type")

        # Initialize the same safe prior in every AHN layer.
        safe_weights = torch.tensor(
            [cfg.safe_bn_weight, cfg.safe_ln_weight, cfg.safe_in_weight],
            dtype=torch.float32,
            device=device,
        )
        safe_weights = safe_weights / safe_weights.sum()

        for m in model.modules():
            if isinstance(m, AdaptiveHybridNorm2d):
                m.set_prior_weights(safe_weights)
        return model

    if method_name == "SN":
        if cfg.model_type == "resnet18":
            return ResNet18_SN(num_classes=cfg.num_classes).to(device)
        if cfg.model_type == "resnet50":
            return ResNet50_SN(num_classes=cfg.num_classes).to(device)
        raise ValueError("Invalid model_type")

    if method_name == "BN":
        if cfg.model_type == "resnet18":
            return ResNet18_BN(num_classes=cfg.num_classes).to(device)
        if cfg.model_type == "resnet50":
            return ResNet50_BN(num_classes=cfg.num_classes).to(device)
        raise ValueError("Invalid model_type")

    if method_name == "GN":
        if cfg.model_type == "resnet18":
            return ResNet18_GN(num_classes=cfg.num_classes, num_groups=cfg.gn_groups).to(device)
        if cfg.model_type == "resnet50":
            return ResNet50_GN(num_classes=cfg.num_classes, num_groups=cfg.gn_groups).to(device)
        raise ValueError("Invalid model_type")

    raise ValueError(f"Unsupported method: {method_name}")


# =============================
# Clean computational-cost runner
# =============================
def run_one_method(method_name: str, cfg: Config, train_loader: DataLoader, device: torch.device) -> Dict[str, object]:
    print("\n" + "=" * 78)
    print(f"STARTING METHOD: {method_name}")
    print("=" * 78)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Re-seed before each method for controlled initialization/data order.
    set_seed(cfg.seed)

    model = build_model(method_name, cfg, device)
    total_params = count_parameters(model)
    forward_macs_per_image = estimate_forward_macs(model, device, input_shape=(1, 3, 32, 32))
    train_images = len(train_loader.dataset)
    macs_per_epoch = int(forward_macs_per_image * TRAINING_BACKWARD_MULTIPLIER * train_images)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    milestones = get_scaled_milestones(cfg.num_epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    method_config = {
        "Method": method_name,
        "Dataset": cfg.dataset_name,
        "Num_Classes": cfg.num_classes,
        "Model_Type": cfg.model_type,
        "Batch_Size": cfg.batch_size,
        "Num_Epochs": cfg.num_epochs,
        "Learning_Rate": cfg.lr,
        "Momentum": cfg.momentum,
        "Weight_Decay": cfg.weight_decay,
        "Scheduler_Milestones": milestones,
        "Parameter_Count": total_params,
        "Forward_MACs_per_Image": forward_macs_per_image,
        "Training_MACs_per_Epoch_Estimate": macs_per_epoch,
    }

    if method_name == "PA-AHN":
        method_config.update({
            "Temperature": cfg.temperature if cfg.use_temperature_scaling else 1.0,
            "Use_LStab": cfg.use_lstab,
            "LStab_Lambda": cfg.lstab_lambda_max if cfg.use_lstab else 0.0,
            "Safe_BN_Weight": cfg.safe_bn_weight,
            "Safe_LN_Weight": cfg.safe_ln_weight,
            "Safe_IN_Weight": cfg.safe_in_weight,
        })
    elif method_name == "SN":
        method_config.update({
            "Use_SN_Batch_Average_Inference_Calibration": cfg.use_sn_batch_average_inference_calibration,
            "SN_Calibration_Batches": cfg.sn_batch_average_calibration_batches,
        })
    elif method_name == "GN":
        method_config.update({"GN_Groups": cfg.gn_groups})

    print("Configuration:")
    print(json.dumps(method_config, indent=2))
    print(f"{method_name} Parameter count: {total_params}")

    epoch_times: List[float] = []
    avg_batch_times: List[float] = []
    method_start_time = time.perf_counter()

    # Clean training loop:
    # - no train/test accuracy calculation
    # - no per-batch weight collection
    # - no .cpu().numpy() logging
    # - no CSV/plot/checkpoint/file logging
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        lambda_lstab = get_lambda_lstab(epoch, cfg.num_epochs, cfg) if method_name == "PA-AHN" else 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            task_loss = criterion(outputs, labels)

            if method_name == "PA-AHN" and cfg.use_lstab:
                entropy_loss, _ = compute_ahn_entropy_loss(model)
                loss = task_loss + lambda_lstab * entropy_loss
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        if method_name == "SN" and cfg.use_sn_batch_average_inference_calibration:
            calibrate_sn_batch_average(
                model=model,
                train_loader=train_loader,
                device=device,
                max_batches=cfg.sn_batch_average_calibration_batches,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        avg_batch_time = epoch_time / max(1, len(train_loader))
        epoch_times.append(float(epoch_time))
        avg_batch_times.append(float(avg_batch_time))

        print(
            f"{method_name} | Epoch [{epoch}/{cfg.num_epochs}] | "
            f"EpochTime: {epoch_time:.2f}s | AvgBatchTime: {avg_batch_time:.6f}s"
        )

    total_training_time_sec = float(time.perf_counter() - method_start_time)
    peak_gpu_memory_mb = get_gpu_memory_mb() if cfg.measure_gpu_memory else 0.0
    average_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    average_batch_time = float(np.mean(avg_batch_times)) if avg_batch_times else 0.0

    result = {
        "Dataset": cfg.dataset_name.upper(),
        "Model": cfg.model_type.upper(),
        "Batch Size": cfg.batch_size,
        "Method": method_name,
        "Parameter Count": total_params,
        "Extra Parameters": 0,  # filled after BN baseline is available
        "FLOPs / MACs per Forward Pass": forward_macs_per_image,
        "FLOPs / MACs per Epoch": macs_per_epoch,
        "Mathematical Operation Impact": "Baseline",  # filled after BN baseline is available
        "Runtime Impact": "Baseline",  # filled after BN baseline is available
        "Peak GPU Memory": peak_gpu_memory_mb,
        "_Runtime_Sec_For_Impact": total_training_time_sec,
    }

    print("-" * 78)
    print(f"{method_name} CLEAN COMPUTATIONAL COST SUMMARY")
    print(f"Parameter count:       {result['Parameter Count']}")
    print(f"Average epoch time:    {average_epoch_time:.2f}s")
    print(f"Average batch time:    {average_batch_time:.6f}s")
    print(f"Peak GPU memory:       {result['Peak GPU Memory']:.2f} MB")
    print("-" * 78)

    # Free memory before next method.
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result



# =============================
# Multi-configuration runner
# =============================
METHODS_TO_RUN = ["PA-AHN", "SN", "BN", "GN"]

EXPERIMENTS = [
    ("cifar10",  "resnet18", 64),
    ("cifar100", "resnet18", 64),
    ("cifar10",  "resnet50", 64),
    ("cifar100", "resnet50", 64),
    ("cifar10",  "resnet18", 128),
    ("cifar100", "resnet18", 128),
    ("cifar10",  "resnet50", 128),
    ("cifar100", "resnet50", 128),
]


def make_cfg(dataset_name: str, model_type: str, batch_size: int) -> Config:
    """Create a fresh config for one dataset/model/batch-size combination."""
    return Config(
        dataset_name=dataset_name,
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=1,  # User requested single epoch for every test.
    )


GEC_COLUMNS = [
    "Dataset",
    "Model",
    "Batch Size",
    "Method",
    "Parameter Count",
    "Extra Parameters",
    "FLOPs / MACs per Forward Pass",
    "FLOPs / MACs per Epoch",
    "Mathematical Operation Impact",
    "Runtime Impact",
    "Peak GPU Memory",
]


def finalize_bn_relative_results(raw_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Fill BN-relative extra parameters, mathematical impact, and runtime impact."""
    bn = next((r for r in raw_results if r["Method"] == "BN"), None)
    if bn is None:
        raise RuntimeError("BN baseline missing; cannot calculate BN-relative impacts.")

    baseline_params = float(bn["Parameter Count"])
    baseline_macs = float(bn["FLOPs / MACs per Epoch"])
    baseline_runtime = float(bn["_Runtime_Sec_For_Impact"])

    finalized = []
    for r in raw_results:
        row = dict(r)
        row["Extra Parameters"] = int(row["Parameter Count"] - baseline_params)
        if row["Method"] == "BN":
            row["Mathematical Operation Impact"] = "Baseline (1.000x, 0.00% vs BN)"
            row["Runtime Impact"] = "Baseline (1.000x, 0.00% vs BN)"
        else:
            row["Mathematical Operation Impact"] = format_impact(float(row["FLOPs / MACs per Epoch"]), baseline_macs)
            row["Runtime Impact"] = format_impact(float(row["_Runtime_Sec_For_Impact"]), baseline_runtime)
        finalized.append({col: row[col] for col in GEC_COLUMNS})
    return finalized


def append_results_to_csv(results: List[Dict[str, object]], csv_path: str) -> None:
    """Append finalized GEC columns for one dataset/model/batch group."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(results, columns=GEC_COLUMNS)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_csv_path = os.path.join(CFG.base_dir, "results", CFG.output_csv_name)

    # Start a fresh CSV for this complete 8-config run.
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    all_results: List[Dict[str, object]] = []

    for exp_idx, (dataset_name, model_type, batch_size) in enumerate(EXPERIMENTS, start=1):
        cfg = make_cfg(dataset_name, model_type, batch_size)
        set_seed(cfg.seed)

        print("\n" + "#" * 90)
        print(
            f"CONFIG {exp_idx}/{len(EXPERIMENTS)} | "
            f"Dataset={cfg.dataset_name.upper()} | "
            f"Model={cfg.model_type.upper()} | "
            f"Batch Size={cfg.batch_size} | "
            f"Epochs={cfg.num_epochs} | LR={cfg.lr}"
        )
        print("#" * 90)

        train_loader, _ = get_dataloaders(cfg)

        raw_config_results: List[Dict[str, object]] = []
        for method_name in METHODS_TO_RUN:
            result = run_one_method(method_name, cfg, train_loader, device)
            raw_config_results.append(result)

        finalized_config_results = finalize_bn_relative_results(raw_config_results)
        append_results_to_csv(finalized_config_results, output_csv_path)
        all_results.extend(finalized_config_results)
        print(f"Saved finalized BN-relative GEC columns to CSV: {output_csv_path}")

        # Release dataloader references before moving to the next configuration.
        del train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print("\n" + "=" * 90)
    print("ALL REQUESTED COMPUTATIONAL-COST TESTS FINISHED")
    print(f"Total rows saved: {len(all_results)}")
    print(f"Final CSV: {output_csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
