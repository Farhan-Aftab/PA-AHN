# ================================================================
# FILE 1: sn_normalization.py
# Training / evaluation / logging for SN on CIFAR-10 / CIFAR-100
# Paper-faithful SN experiment structure
# Includes:
# - statistic-level SN
# - paper-style step decay (scaled to total epochs)
# - optional batch-average inference calibration
# - training time per epoch
# - total training time
# - time per batch
# - parameter count
# - GPU memory usage
# - best test accuracy
# ================================================================

import os
import sys
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet18_sn import ResNet18_SN
from models.resnet50_sn import ResNet50_SN
from norms.switchable_norm import SwitchableNorm2d


# =============================
# Configuration
# =============================
@dataclass
class Config:
    base_dir: str = "./runs"

    dataset_name: str = "cifar10"
    num_classes: int = 10

    # dataset_name: str = "cifar100"
    # num_classes: int = 100

    model_type: str = "resnet18"
    # model_type: str = "resnet50"

    batch_size: int = 8                 # options: batch_size 8
    # batch_size: int = 16                # options: batch_size 16
    # batch_size: int = 32                # options: batch_size 32
    # batch_size: int = 64                # options: batch_size 64
    # batch_size: int = 128               # options: batch_size 128

    num_epochs: int = 100                # option: epochs 100
    # num_epochs: int = 50                # option: epochs 50
    # num_epochs: int = 20                # option: epochs 20

    num_workers: int = 2

    momentum: float = 0.9

    weight_decay: float = 1e-4   # paper-style weight decay
    # weight_decay: float = 5e-4   # paper-style weight decay

    seed: int = 42

    # Optional entropy logging only, no L-Stab in baseline
    log_entropy: bool = True

    # Paper-style step decay:
    # original paper uses 100 epochs and drops at 30, 60, 90.
    # Here we scale those milestones proportionally to total epochs.
    use_paper_style_scaled_milestones: bool = True

    # Batch-average inference calibration:
    # original SN paper uses batch average rather than moving average in test.

    use_batch_average_inference: bool = True
    inference: str = "Y"

    # use_batch_average_inference: bool = False
    # inference: str = "N"

    # None = use full training loader for calibration
    # For speed, user may set a smaller integer later if needed.
    batch_average_calibration_batches: Optional[int] = None

    # time / memory measurement
    measure_batch_time: bool = True
    measure_gpu_memory: bool = True


CFG = Config()


# Paper-style SN learning-rate scaling
# SN repo reference:
# effective batch 256 -> lr 0.1
# effective batch 32  -> lr 0.0125
# effective batch 8   -> lr 0.003125
# rest to values like for 128, 64  and 16 I derived them by linear scaling from the SN repo’s reported ImageNet settings.
# For our single-GPU CIFAR experiments:
# Therefore:
# lr = 0.1 * CFG.batch_size / 256
CFG.lr = 0.1 * CFG.batch_size / 256


# Experiment naming
CFG.experiment_name = f"sn_paperfaithful_{CFG.dataset_name.lower()}_{CFG.model_type}_batchsize_{CFG.batch_size}_lr_{CFG.lr}_wd_{CFG.weight_decay}_inf_{CFG.inference}_{CFG.num_epochs}epochs"

# CFG.experiment_name = f"sn_paperfaithful_{CFG.dataset_name.lower()}_{CFG.model_type}_{CFG.batch_size}_{CFG.num_epochs}epochs"

# Add base dir to path
sys.path.append(CFG.base_dir)


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


def append_to_results_file(file_path: str, text: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())


def ensure_dirs(base_dir: str) -> Dict[str, str]:
    paths = {
        "weights": os.path.join(base_dir, "weights"),
        "plots": os.path.join(base_dir, "plots"),
        "models_ckpt": os.path.join(base_dir, "models_ckpt"),
        "results": os.path.join(base_dir, "results"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


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
    else:
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def compute_sn_entropy(model: nn.Module) -> Tuple[torch.Tensor, int]:
    entropy_loss = None
    sn_layers = 0

    for m in model.modules():
        if isinstance(m, SwitchableNorm2d):
            layer_entropy = m.compute_entropy()
            if entropy_loss is None:
                entropy_loss = layer_entropy
            else:
                entropy_loss = entropy_loss + layer_entropy
            sn_layers += 1

    if entropy_loss is None:
        device = next(model.parameters()).device
        entropy_loss = torch.tensor(0.0, device=device)
    elif sn_layers > 0:
        entropy_loss = entropy_loss / sn_layers

    return entropy_loss, sn_layers


def collect_sn_weights(model: nn.Module) -> np.ndarray:
    weights = []
    for m in model.modules():
        if isinstance(m, SwitchableNorm2d) and m.last_weights is not None:
            w = m.last_weights.mean(dim=0).detach().cpu().numpy()
            weights.append(w)

    if len(weights) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    weights = np.array(weights, dtype=np.float32)
    return weights.mean(axis=0)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    avg_acc = 100.0 * correct / total
    return float(avg_loss), float(avg_acc)


def plot_curve(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weight_bars(all_epoch_averages: List[Dict[str, float]], save_path: str, title: str):
    epochs = list(range(1, len(all_epoch_averages) + 1))
    bn_values = [e["BN"] for e in all_epoch_averages]
    ln_values = [e["LN"] for e in all_epoch_averages]
    in_values = [e["IN"] for e in all_epoch_averages]

    width = 0.25
    x = np.arange(len(epochs))

    plt.figure(figsize=(20, 6))
    plt.bar(x - width, bn_values, width, label="BN")
    plt.bar(x, ln_values, width, label="LN")
    plt.bar(x + width, in_values, width, label="IN")
    plt.xlabel("Epoch")
    plt.ylabel("Average SN Weight")
    plt.title(title)
    plt.xticks(x, [f"{e}" for e in epochs])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


def get_scaled_milestones(num_epochs: int) -> List[int]:
    """
    Scale the paper's [30, 60, 90] over 100 epochs proportionally.
    """
    ratios = [0.30, 0.60, 0.90]
    milestones = sorted(set(max(1, int(round(num_epochs * r))) for r in ratios if int(round(num_epochs * r)) < num_epochs))
    return milestones


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


#def calibrate_sn_batch_average(model: nn.Module, train_loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> None:
    """
    Collect batch-average BN statistics from the training set
    for paper-style SN inference.
    """
#    model.eval()
#    start_batch_average_collection(model)

#   with torch.no_grad():
#        for batch_idx, (inputs, _) in enumerate(train_loader):
#            if max_batches is not None and batch_idx >= max_batches:
#                break
#            inputs = inputs.to(device)
#            _ = model(inputs)

#    stop_batch_average_collection(model)
#    finalize_batch_average(model)


def calibrate_sn_batch_average(model: nn.Module, train_loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> None:
    """
    Collect batch-average BN statistics from the training set
    for paper-style SN inference.

    FIX:
    - Must run in train() mode because SN collects BN stats only in training path
    - Still uses torch.no_grad() so no gradients are computed
    - Restores original train/eval mode after calibration
    """
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
# Main training function
# =============================
def main():
    set_seed(CFG.seed)
    paths = ensure_dirs(CFG.base_dir)

    results_txt = os.path.join(
        paths["results"],
        f"{CFG.experiment_name}_PRINT_RESULTS.txt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    line = f"Using device: {device}"
    print(line)
    append_to_results_file(results_txt, line)

    line = "Config:"
    print(line)
    append_to_results_file(results_txt, line)

    config_text = json.dumps(asdict(CFG), indent=2)
    print(config_text)
    append_to_results_file(results_txt, config_text)

    train_loader, test_loader = get_dataloaders(CFG)

    if CFG.model_type == "resnet18":
        model = ResNet18_SN(num_classes=CFG.num_classes).to(device)
    elif CFG.model_type == "resnet50":
        model = ResNet50_SN(num_classes=CFG.num_classes).to(device)
    else:
        raise ValueError("Invalid model_type")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=CFG.lr,
        momentum=CFG.momentum,
        weight_decay=CFG.weight_decay,
    )

    if CFG.use_paper_style_scaled_milestones:
        milestones = get_scaled_milestones(CFG.num_epochs)
    else:
        milestones = [10, 15]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
#    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epochs)


    # =============================
    # New overall metrics
    # =============================
    total_params = count_parameters(model)
    total_trainable_params = total_params
    total_training_start_time = time.time()

    if torch.cuda.is_available() and CFG.measure_gpu_memory:
        torch.cuda.reset_peak_memory_stats(device)

    # Storage
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    entropy_history = []
    epoch_time_history = []
    avg_batch_time_history = []
    gpu_memory_history = []

    train_metrics = {
        "Epoch": [],
        "Train_Loss": [],
        "Train_Accuracy": [],
        "Test_Loss": [],
        "Test_Accuracy": [],
        "Best_Test_Accuracy_So_Far": [],
        "Entropy": [],
        "BN": [],
        "LN": [],
        "IN": [],
        "Epoch_Time_Sec": [],
        "Avg_Batch_Time_Sec": [],
        "GPU_Memory_MB": [],
        "Parameter_Count": [],
    }

    all_epoch_averages = []
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, CFG.num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        entropy_epoch_sum = 0.0
        entropy_batches = 0
        batch_weight_log = []
        batch_time_log = []

        for inputs, labels in train_loader:
            batch_start_time = time.time()

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_w = collect_sn_weights(model)
            batch_weight_log.append(avg_w)

            entropy_loss, _ = compute_sn_entropy(model)
            entropy_epoch_sum += float(entropy_loss.item())
            entropy_batches += 1

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if CFG.measure_batch_time:
                batch_time_log.append(time.time() - batch_start_time)

        scheduler.step()

        # Paper-style batch-average inference calibration
        if CFG.use_batch_average_inference:
            calibrate_sn_batch_average(
                model=model,
                train_loader=train_loader,
                device=device,
                max_batches=CFG.batch_average_calibration_batches,
            )

        epoch_loss = float(running_loss / len(train_loader.dataset))
        epoch_acc = float(100.0 * correct / total)
        avg_entropy = float(entropy_epoch_sum / max(1, entropy_batches))
        epoch_time_sec = float(time.time() - epoch_start_time)
        avg_batch_time_sec = float(np.mean(batch_time_log)) if len(batch_time_log) > 0 else 0.0

        if len(batch_weight_log) > 0:
            batch_weight_log = np.array(batch_weight_log)
            epoch_avg = {
                "BN": float(batch_weight_log[:, 0].mean()),
                "LN": float(batch_weight_log[:, 1].mean()),
                "IN": float(batch_weight_log[:, 2].mean()),
            }
        else:
            epoch_avg = {"BN": 0.0, "LN": 0.0, "IN": 0.0}

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        current_gpu_memory_mb = get_gpu_memory_mb() if CFG.measure_gpu_memory else 0.0

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        entropy_history.append(avg_entropy)
        epoch_time_history.append(epoch_time_sec)
        avg_batch_time_history.append(avg_batch_time_sec)
        gpu_memory_history.append(current_gpu_memory_mb)
        all_epoch_averages.append(epoch_avg)

        train_metrics["Epoch"].append(epoch)
        train_metrics["Train_Loss"].append(epoch_loss)
        train_metrics["Train_Accuracy"].append(epoch_acc)
        train_metrics["Test_Loss"].append(test_loss)
        train_metrics["Test_Accuracy"].append(test_acc)
        train_metrics["Best_Test_Accuracy_So_Far"].append(best_test_acc)
        train_metrics["Entropy"].append(avg_entropy)
        train_metrics["BN"].append(epoch_avg["BN"])
        train_metrics["LN"].append(epoch_avg["LN"])
        train_metrics["IN"].append(epoch_avg["IN"])
        train_metrics["Epoch_Time_Sec"].append(epoch_time_sec)
        train_metrics["Avg_Batch_Time_Sec"].append(avg_batch_time_sec)
        train_metrics["GPU_Memory_MB"].append(current_gpu_memory_mb)
        train_metrics["Parameter_Count"].append(total_params)

        epoch_line = (
            f"Epoch [{epoch}/{CFG.num_epochs}] "
            f"BatchSize:{CFG.batch_size} | "
            f"TrainLoss:{epoch_loss:.4f} TrainAcc:{epoch_acc:.2f}% "
            f"TestLoss:{test_loss:.4f} TestAcc:{test_acc:.2f}% "
            f"BestTestAcc:{best_test_acc:.2f}% "
            f"Entropy:{avg_entropy:.4f} | "
            # f"BN:{epoch_avg['BN']:.3f} LN:{epoch_avg['LN']:.3f} IN:{epoch_avg['IN']:.3f} | "
            f"BN:{epoch_avg['BN']:.6f} LN:{epoch_avg['LN']:.6f} IN:{epoch_avg['IN']:.6f} | "
            f"EpochTime:{epoch_time_sec:.2f}s BatchTime:{avg_batch_time_sec:.4f}s "
            f"GPU_Mem:{current_gpu_memory_mb:.2f}MB"
        )

        print(epoch_line)
        append_to_results_file(results_txt, epoch_line)

        ckpt_path = os.path.join(paths["models_ckpt"], f"{CFG.experiment_name}.pth")
        torch.save(model.state_dict(), ckpt_path)

        if epoch == best_epoch:
            best_path = os.path.join(paths["models_ckpt"], f"{CFG.experiment_name}_best.pth")
            torch.save(model.state_dict(), best_path)

    total_training_time_sec = float(time.time() - total_training_start_time)
    peak_gpu_memory_mb = max(gpu_memory_history) if len(gpu_memory_history) > 0 else 0.0

    with open(os.path.join(paths["results"], f"{CFG.experiment_name}_config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    metrics_csv = os.path.join(paths["results"], f"{CFG.experiment_name}_training_metrics.csv")
    weights_csv = os.path.join(paths["results"], f"{CFG.experiment_name}_avg_weights.csv")
    summary_csv = os.path.join(paths["results"], f"{CFG.experiment_name}_summary_metrics.csv")

    pd.DataFrame(train_metrics).to_csv(metrics_csv, index=False)
    pd.DataFrame(all_epoch_averages).to_csv(weights_csv, index_label="Epoch")

    summary_metrics = {
        "Experiment_Name": [CFG.experiment_name],
        "Model_Type": [CFG.model_type],
        "Dataset": [CFG.dataset_name],
        "Num_Classes": [CFG.num_classes],
        "Num_Epochs": [CFG.num_epochs],
        "Batch_Size": [CFG.batch_size],
        "Scheduler_Milestones": [str(milestones)],
        "Parameter_Count": [total_trainable_params],
        "Best_Test_Accuracy": [best_test_acc],
        "Best_Epoch": [best_epoch],
        "Final_Test_Accuracy": [test_acc_history[-1] if len(test_acc_history) > 0 else 0.0],
        "Final_Test_Loss": [test_loss_history[-1] if len(test_loss_history) > 0 else 0.0],
        "Average_Epoch_Time_Sec": [float(np.mean(epoch_time_history)) if len(epoch_time_history) > 0 else 0.0],
        "Total_Training_Time_Sec": [total_training_time_sec],
        "Average_Batch_Time_Sec": [float(np.mean(avg_batch_time_history)) if len(avg_batch_time_history) > 0 else 0.0],
        "Peak_GPU_Memory_MB": [peak_gpu_memory_mb],
    }
    pd.DataFrame(summary_metrics).to_csv(summary_csv, index=False)

    epochs = list(range(1, CFG.num_epochs + 1))
    plot_curve(
        epochs,
        train_loss_history,
        f"SN Training Loss - {CFG.experiment_name}",
        "Epoch",
        "Loss",
        os.path.join(paths["results"], f"{CFG.experiment_name}_train_loss.png"),
    )
    plot_curve(
        epochs,
        train_acc_history,
        f"SN Training Accuracy - {CFG.experiment_name}",
        "Epoch",
        "Accuracy (%)",
        os.path.join(paths["results"], f"{CFG.experiment_name}_train_accuracy.png"),
    )
    plot_curve(
        epochs,
        test_acc_history,
        f"SN Test Accuracy - {CFG.experiment_name}",
        "Epoch",
        "Accuracy (%)",
        os.path.join(paths["results"], f"{CFG.experiment_name}_test_accuracy.png"),
    )
    plot_curve(
        epochs,
        entropy_history,
        f"SN Entropy - {CFG.experiment_name}",
        "Epoch",
        "Entropy",
        os.path.join(paths["results"], f"{CFG.experiment_name}_entropy.png"),
    )
    plot_curve(
        epochs,
        epoch_time_history,
        f"SN Training Time per Epoch - {CFG.experiment_name}",
        "Epoch",
        "Time (sec)",
        os.path.join(paths["results"], f"{CFG.experiment_name}_epoch_time.png"),
    )
    plot_curve(
        epochs,
        avg_batch_time_history,
        f"SN Average Batch Time - {CFG.experiment_name}",
        "Epoch",
        "Time (sec)",
        os.path.join(paths["results"], f"{CFG.experiment_name}_batch_time.png"),
    )
    plot_curve(
        epochs,
        gpu_memory_history,
        f"SN GPU Memory Usage - {CFG.experiment_name}",
        "Epoch",
        "Memory (MB)",
        os.path.join(paths["results"], f"{CFG.experiment_name}_gpu_memory.png"),
    )
    plot_weight_bars(
        all_epoch_averages,
        os.path.join(paths["results"], f"{CFG.experiment_name}_avg_weights.png"),
        f"Average SN Weights Across Epochs - {CFG.experiment_name}",
    )

    final_lines = [
        "Training finished.",
        f"SN Parameter count: {total_trainable_params}",
        f"SN Best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}",
        f"SN Final test accuracy: {test_acc_history[-1]:.2f}%" if len(test_acc_history) > 0 else "Final test accuracy: N/A",
        f"SN Average epoch time: {np.mean(epoch_time_history):.2f}s" if len(epoch_time_history) > 0 else "Average epoch time: N/A",
        f"SN Total training time: {total_training_time_sec:.2f}s",
        f"SN Average batch time: {np.mean(avg_batch_time_history):.4f}s" if len(avg_batch_time_history) > 0 else "Average batch time: N/A",
        f"SN Peak GPU memory: {peak_gpu_memory_mb:.2f} MB",
        f"SN Metrics saved to: {metrics_csv}",
        f"SN Summary saved to: {summary_csv}",
        f"SN Weights saved to: {weights_csv}",
    ]

    for line in final_lines:
        print(line)
        append_to_results_file(results_txt, line)


if __name__ == "__main__":
    main()
