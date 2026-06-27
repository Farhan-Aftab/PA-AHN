# ================================================================
# FILE 1: gn_normalization.py
# Training / evaluation / logging for Group Normalization baseline
# on CIFAR-10 / CIFAR-100
# Journal-support baseline for PA-AHN vs SN comparison
# Keeps comparable logging fields used in SN / PA-AHN experiments:
# - training / test loss and accuracy
# - training time per epoch
# - total training time
# - average time per batch
# - parameter count
# - GPU memory usage
# - best test accuracy
# - CSV summaries and plots
# Note:
# - Entropy is logged as 0.0 because GN has no adaptive branch routing.
# - BN/LN/IN/GN indicator columns are logged for comparison tables.
# ================================================================

import os
import sys
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet18_gn import ResNet18_GN
from models.resnet50_gn import ResNet50_GN


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

    # model_type: str = "resnet18"
    model_type: str = "resnet50"

    # batch_size: int = 8                 # options: 8, 16, 32, 64, 128, 256
    # batch_size: int = 32
    batch_size: int = 64
    # batch_size: int = 128
    # batch_size: int = 256

    num_epochs: int = 100               # final paper/thesis setting
    # num_epochs: int = 50
    # num_epochs: int = 20

    num_workers: int = 2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42

    # GN setting: 32 is standard for ResNet channels 64/128/256/512.
    # If channels are not divisible by 32, code automatically falls back
    # to the largest valid divisor.
    gn_groups: int = 32

    # Same scheduler policy as SN / PA-AHN experiments
    use_paper_style_scaled_milestones: bool = True

    # time / memory measurement
    measure_batch_time: bool = True
    measure_gpu_memory: bool = True

    baseline_name: str = "GN"


CFG = Config()

# Same paper-style linear learning-rate scaling used for SN / PA-AHN fairness
CFG.lr = 0.1 * CFG.batch_size / 256

CFG.experiment_name = (
    f"gn_baseline_{CFG.dataset_name.lower()}_{CFG.model_type}_"
    f"batchsize_{CFG.batch_size}_lr_{CFG.lr}_wd_{CFG.weight_decay}_"
    f"groups_{CFG.gn_groups}_{CFG.num_epochs}epochs"
)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
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


def plot_norm_indicator_bars(all_epoch_averages: List[Dict[str, float]], save_path: str, title: str):
    epochs = list(range(1, len(all_epoch_averages) + 1))
    bn_values = [e["BN"] for e in all_epoch_averages]
    ln_values = [e["LN"] for e in all_epoch_averages]
    in_values = [e["IN"] for e in all_epoch_averages]
    gn_values = [e["GN"] for e in all_epoch_averages]

    width = 0.20
    x = np.arange(len(epochs))

    plt.figure(figsize=(20, 6))
    plt.bar(x - 1.5 * width, bn_values, width, label="BN")
    plt.bar(x - 0.5 * width, ln_values, width, label="LN")
    plt.bar(x + 0.5 * width, in_values, width, label="IN")
    plt.bar(x + 1.5 * width, gn_values, width, label="GN")
    plt.xlabel("Epoch")
    plt.ylabel("Normalization Indicator")
    plt.title(title)
    plt.xticks(x, [f"{e}" for e in epochs])
    plt.ylim(0, 1.05)
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
    ratios = [0.30, 0.60, 0.90]
    milestones = sorted(set(max(1, int(round(num_epochs * r))) for r in ratios if int(round(num_epochs * r)) < num_epochs))
    return milestones


def get_baseline_norm_indicator() -> Dict[str, float]:
    return {"BN": 0.0, "LN": 0.0, "IN": 0.0, "GN": 1.0}


# =============================
# Main training function
# =============================
def main():
    set_seed(CFG.seed)
    paths = ensure_dirs(CFG.base_dir)

    results_txt = os.path.join(paths["results"], f"{CFG.experiment_name}_PRINT_RESULTS.txt")
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
        model = ResNet18_GN(num_classes=CFG.num_classes, num_groups=CFG.gn_groups).to(device)
    elif CFG.model_type == "resnet50":
        model = ResNet50_GN(num_classes=CFG.num_classes, num_groups=CFG.gn_groups).to(device)
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

    total_params = count_parameters(model)
    total_trainable_params = total_params
    total_training_start_time = time.time()

    if torch.cuda.is_available() and CFG.measure_gpu_memory:
        torch.cuda.reset_peak_memory_stats(device)

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
        "GN": [],
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
        batch_time_log = []

        for inputs, labels in train_loader:
            batch_start_time = time.time()

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if CFG.measure_batch_time:
                batch_time_log.append(time.time() - batch_start_time)

        scheduler.step()

        epoch_loss = float(running_loss / len(train_loader.dataset))
        epoch_acc = float(100.0 * correct / total)
        avg_entropy = 0.0
        epoch_avg = get_baseline_norm_indicator()
        epoch_time_sec = float(time.time() - epoch_start_time)
        avg_batch_time_sec = float(np.mean(batch_time_log)) if len(batch_time_log) > 0 else 0.0

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
        train_metrics["GN"].append(epoch_avg["GN"])
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
            f"BN:{epoch_avg['BN']:.6f} LN:{epoch_avg['LN']:.6f} "
            f"IN:{epoch_avg['IN']:.6f} GN:{epoch_avg['GN']:.6f} | "
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
        "Baseline": [CFG.baseline_name],
        "Model_Type": [CFG.model_type],
        "Dataset": [CFG.dataset_name],
        "Num_Classes": [CFG.num_classes],
        "Num_Epochs": [CFG.num_epochs],
        "Batch_Size": [CFG.batch_size],
        "Learning_Rate": [CFG.lr],
        "Weight_Decay": [CFG.weight_decay],
        "GN_Groups": [CFG.gn_groups],
        "Scheduler_Milestones": [str(milestones)],
        "Parameter_Count": [total_trainable_params],
        "Best_Test_Accuracy": [best_test_acc],
        "Best_Epoch": [best_epoch],
        "Final_Test_Accuracy": [test_acc_history[-1] if len(test_acc_history) > 0 else 0.0],
        "Final_Test_Loss": [test_loss_history[-1] if len(test_loss_history) > 0 else 0.0],
        "Final_Entropy": [entropy_history[-1] if len(entropy_history) > 0 else 0.0],
        "Final_BN": [all_epoch_averages[-1]["BN"] if len(all_epoch_averages) > 0 else 0.0],
        "Final_LN": [all_epoch_averages[-1]["LN"] if len(all_epoch_averages) > 0 else 0.0],
        "Final_IN": [all_epoch_averages[-1]["IN"] if len(all_epoch_averages) > 0 else 0.0],
        "Final_GN": [all_epoch_averages[-1]["GN"] if len(all_epoch_averages) > 0 else 0.0],
        "Average_Epoch_Time_Sec": [float(np.mean(epoch_time_history)) if len(epoch_time_history) > 0 else 0.0],
        "Total_Training_Time_Sec": [total_training_time_sec],
        "Average_Batch_Time_Sec": [float(np.mean(avg_batch_time_history)) if len(avg_batch_time_history) > 0 else 0.0],
        "Peak_GPU_Memory_MB": [peak_gpu_memory_mb],
    }
    pd.DataFrame(summary_metrics).to_csv(summary_csv, index=False)

    epochs = list(range(1, CFG.num_epochs + 1))
    plot_curve(epochs, train_loss_history, f"GN Training Loss - {CFG.experiment_name}", "Epoch", "Loss", os.path.join(paths["results"], f"{CFG.experiment_name}_train_loss.png"))
    plot_curve(epochs, train_acc_history, f"GN Training Accuracy - {CFG.experiment_name}", "Epoch", "Accuracy (%)", os.path.join(paths["results"], f"{CFG.experiment_name}_train_accuracy.png"))
    plot_curve(epochs, test_acc_history, f"GN Test Accuracy - {CFG.experiment_name}", "Epoch", "Accuracy (%)", os.path.join(paths["results"], f"{CFG.experiment_name}_test_accuracy.png"))
    plot_curve(epochs, entropy_history, f"GN Entropy Indicator - {CFG.experiment_name}", "Epoch", "Entropy", os.path.join(paths["results"], f"{CFG.experiment_name}_entropy.png"))
    plot_curve(epochs, epoch_time_history, f"GN Training Time per Epoch - {CFG.experiment_name}", "Epoch", "Time (sec)", os.path.join(paths["results"], f"{CFG.experiment_name}_epoch_time.png"))
    plot_curve(epochs, avg_batch_time_history, f"GN Average Batch Time - {CFG.experiment_name}", "Epoch", "Time (sec)", os.path.join(paths["results"], f"{CFG.experiment_name}_batch_time.png"))
    plot_curve(epochs, gpu_memory_history, f"GN GPU Memory Usage - {CFG.experiment_name}", "Epoch", "Memory (MB)", os.path.join(paths["results"], f"{CFG.experiment_name}_gpu_memory.png"))
    plot_norm_indicator_bars(all_epoch_averages, os.path.join(paths["results"], f"{CFG.experiment_name}_norm_indicators.png"), f"GN Normalization Indicator Across Epochs - {CFG.experiment_name}")

    final_lines = [
        "Training finished.",
        f"GN Parameter count: {total_trainable_params}",
        f"GN Best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}",
        f"GN Final test accuracy: {test_acc_history[-1]:.2f}%" if len(test_acc_history) > 0 else "Final test accuracy: N/A",
        f"GN Average epoch time: {np.mean(epoch_time_history):.2f}s" if len(epoch_time_history) > 0 else "Average epoch time: N/A",
        f"GN Total training time: {total_training_time_sec:.2f}s",
        f"GN Average batch time: {np.mean(avg_batch_time_history):.4f}s" if len(avg_batch_time_history) > 0 else "Average batch time: N/A",
        f"GN Peak GPU memory: {peak_gpu_memory_mb:.2f} MB",
        f"GN Metrics saved to: {metrics_csv}",
        f"GN Summary saved to: {summary_csv}",
        f"GN Indicator weights saved to: {weights_csv}",
    ]

    for line in final_lines:
        print(line)
        append_to_results_file(results_txt, line)


if __name__ == "__main__":
    main()
