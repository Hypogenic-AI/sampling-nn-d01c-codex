"""Run intermediate activation sampling experiments.

This script trains compact CNNs on local HuggingFace datasets and evaluates
whether sampling at a hidden activation layer changes behavior compared with
deterministic inference, final-softmax sampling, Gaussian activation noise,
SAP-style activation sampling, and MC dropout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


@dataclass(frozen=True)
class ExperimentConfig:
    datasets: tuple[str, ...] = ("mnist", "fashion_mnist")
    seeds: tuple[int, ...] = (42, 123, 456)
    train_size: int = 20_000
    val_size: int = 5_000
    test_size: int | None = None
    batch_size: int = 256
    epochs: int = 8
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout_p: float = 0.25
    categorical_keep_fraction: float = 0.25
    sap_keep_fraction: float = 0.35
    gaussian_sigma: float = 0.35
    categorical_tau: float = 0.75
    sample_counts: tuple[int, ...] = (1, 5, 20)
    ece_bins: int = 10
    use_amp: bool = True


class SmallCNN(nn.Module):
    """Compact CNN exposing the penultimate activation for interventions."""

    def __init__(self, hidden_dim: int = 128, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(
        self,
        x: torch.Tensor,
        hidden_intervention: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        hidden = F.relu(self.fc1(x))
        if hidden_intervention is not None:
            hidden = hidden_intervention(hidden)
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pil_images_to_tensor(split, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for idx in indices:
        row = split[int(idx)]
        arr = np.asarray(row["image"].convert("L"), dtype=np.float32) / 255.0
        images.append(arr[None, :, :])
        labels.append(int(row["label"]))
    x = torch.from_numpy(np.stack(images, axis=0)).float()
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def stratified_take(
    indices: np.ndarray, labels: np.ndarray, size: int | None, seed: int
) -> np.ndarray:
    if size is None or size >= len(indices):
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=size,
        stratify=labels[indices],
        random_state=seed,
    )
    return np.asarray(selected)


def prepare_data(
    dataset_name: str, seed: int, cfg: ExperimentConfig
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    ds = load_from_disk(str(ROOT / "datasets" / dataset_name))
    train_labels = np.asarray(ds["train"]["label"], dtype=np.int64)
    all_train_indices = np.arange(len(train_labels))

    requested = min(cfg.train_size + cfg.val_size, len(all_train_indices))
    selected = stratified_take(all_train_indices, train_labels, requested, seed)
    train_idx, val_idx = train_test_split(
        selected,
        train_size=cfg.train_size,
        test_size=cfg.val_size,
        stratify=train_labels[selected],
        random_state=seed,
    )

    test_labels = np.asarray(ds["test"]["label"], dtype=np.int64)
    test_idx = stratified_take(np.arange(len(test_labels)), test_labels, cfg.test_size, seed)

    x_train, y_train = pil_images_to_tensor(ds["train"], train_idx)
    x_val, y_val = pil_images_to_tensor(ds["train"], val_idx)
    x_test, y_test = pil_images_to_tensor(ds["test"], test_idx)

    mean = x_train.mean()
    std = x_train.std().clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )

    metadata = {
        "dataset": dataset_name,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "mean": float(mean.item()),
        "std": float(std.item()),
        "train_class_counts": np.bincount(y_train.numpy(), minlength=10).tolist(),
        "val_class_counts": np.bincount(y_val.numpy(), minlength=10).tolist(),
        "test_class_counts": np.bincount(y_test.numpy(), minlength=10).tolist(),
    }
    return train_loader, val_loader, test_loader, metadata


def model_checkpoint_path(dataset: str, seed: int, variant: str) -> Path:
    return CHECKPOINT_DIR / f"{dataset}_seed{seed}_{variant}.pt"


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += float(loss.item()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return total_loss / total, correct / total


def train_model(
    dataset: str,
    seed: int,
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[SmallCNN, list[dict]]:
    set_seed(seed)
    dropout_p = cfg.dropout_p if variant == "dropout" else 0.0
    model = SmallCNN(hidden_dim=cfg.hidden_dim, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    patience_left = cfg.patience
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item()) * y.numel()
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_total += int(y.numel())

        val_loss, val_acc = evaluate_loss(model, val_loader, device, cfg.use_amp)
        row = {
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(row)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError(f"No model state saved for {dataset} seed {seed} {variant}")

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "config": asdict(cfg),
        "history": history,
    }
    torch.save(checkpoint, model_checkpoint_path(dataset, seed, variant))
    return model, history


def hidden_categorical_intervention(
    h: torch.Tensor, keep_fraction: float, tau: float
) -> torch.Tensor:
    abs_h = h.detach().abs() + 1e-8
    probs = F.softmax(abs_h / tau, dim=1)
    dims = h.shape[1]
    draws = max(1, int(round(keep_fraction * dims)))
    sampled = torch.multinomial(probs, num_samples=draws, replacement=True)
    counts = torch.zeros_like(h)
    counts.scatter_add_(1, sampled, torch.ones_like(sampled, dtype=h.dtype))
    scale = counts / (draws * probs.clamp_min(1e-6))
    return h * scale


def sap_intervention(h: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    weights = h.detach().abs() + 1e-8
    dims = h.shape[1]
    probs = keep_fraction * dims * weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probs = probs.clamp(min=1e-4, max=0.95)
    mask = torch.bernoulli(probs)
    return h * mask / probs


def gaussian_intervention(h: torch.Tensor, sigma: float) -> torch.Tensor:
    per_item_std = h.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
    return h + torch.randn_like(h) * sigma * per_item_std


def enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    prob_variance: float,
    ece_bins: int,
) -> dict:
    eps = 1e-12
    labels = labels.astype(np.int64)
    pred = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1], dtype=np.float32)[labels]
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    cm = confusion_matrix(labels, pred, labels=list(range(probs.shape[1])))
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "accuracy": float((pred == labels).mean()),
        "nll": float(nll.mean()),
        "ece": expected_calibration_error(probs, labels, bins=ece_bins),
        "brier": float(brier),
        "entropy": float(entropy.mean()),
        "prob_variance": float(prob_variance),
        "per_class_accuracy_mean": float(per_class_acc.mean()),
        "per_class_accuracy_min": float(per_class_acc.min()),
    }


def evaluate_method(
    model: SmallCNN,
    loader: DataLoader,
    device: torch.device,
    method: str,
    sample_count: int,
    cfg: ExperimentConfig,
) -> tuple[dict, np.ndarray, np.ndarray]:
    all_sample_probs: list[np.ndarray] = []
    labels_out: np.ndarray | None = None
    start = time.perf_counter()

    with torch.no_grad():
        for sample_idx in range(sample_count):
            batch_probs: list[np.ndarray] = []
            batch_labels: list[np.ndarray] = []

            if method == "mc_dropout":
                enable_dropout_only(model)
            else:
                model.eval()

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                intervention = None
                if method == "hidden_categorical":
                    intervention = lambda h: hidden_categorical_intervention(
                        h, cfg.categorical_keep_fraction, cfg.categorical_tau
                    )
                elif method == "sap":
                    intervention = lambda h: sap_intervention(h, cfg.sap_keep_fraction)
                elif method == "gaussian":
                    intervention = lambda h: gaussian_intervention(h, cfg.gaussian_sigma)

                logits = model(x, hidden_intervention=intervention)
                probs = F.softmax(logits, dim=1)

                if method == "output_sample":
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                    probs = F.one_hot(sampled, num_classes=probs.shape[1]).float()

                batch_probs.append(probs.detach().cpu().numpy())
                if sample_idx == 0:
                    batch_labels.append(y.detach().cpu().numpy())

            all_sample_probs.append(np.concatenate(batch_probs, axis=0))
            if sample_idx == 0:
                labels_out = np.concatenate(batch_labels, axis=0)

    if labels_out is None:
        raise RuntimeError("Evaluation produced no labels")

    stack = np.stack(all_sample_probs, axis=0)
    mean_probs = stack.mean(axis=0)
    prob_variance = float(stack.var(axis=0).mean()) if sample_count > 1 else 0.0
    metrics = compute_metrics(mean_probs, labels_out, prob_variance, cfg.ece_bins)
    metrics["eval_seconds"] = time.perf_counter() - start
    metrics["sample_count"] = sample_count
    metrics["method"] = method
    return metrics, mean_probs, labels_out


def method_grid() -> list[tuple[str, str, Iterable[int]]]:
    return [
        ("deterministic", "deterministic", (1,)),
        ("deterministic", "output_sample", (1, 5, 20)),
        ("deterministic", "hidden_categorical", (1, 5, 20)),
        ("deterministic", "sap", (1, 5, 20)),
        ("deterministic", "gaussian", (1, 5, 20)),
        ("dropout", "dropout_eval", (1,)),
        ("dropout", "mc_dropout", (1, 5, 20)),
    ]


def primary_sample_count(method: str, cfg: ExperimentConfig) -> int:
    if method in {"deterministic", "dropout_eval"}:
        return 1
    return max(cfg.sample_counts)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "nll",
        "ece",
        "brier",
        "entropy",
        "prob_variance",
        "eval_seconds",
        "per_class_accuracy_min",
    ]
    grouped = metrics_df.groupby(["dataset", "model_variant", "method", "sample_count"])
    rows = []
    for keys, group in grouped:
        row = dict(zip(["dataset", "model_variant", "method", "sample_count"], keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            lo, hi = mean_ci(group[metric].to_numpy())
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "model_variant", "method", "sample_count"]
    )


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    mean = values.mean()
    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return float(mean - margin), float(mean + margin)


def cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def statistical_tests(metrics_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    primary_rows = []
    metrics_to_test = ["accuracy", "nll", "ece", "entropy", "prob_variance"]

    for dataset, dataset_df in metrics_df.groupby("dataset"):
        det = dataset_df[
            (dataset_df["model_variant"] == "deterministic")
            & (dataset_df["method"] == "deterministic")
            & (dataset_df["sample_count"] == 1)
        ]
        for method in ["output_sample", "hidden_categorical", "sap", "gaussian"]:
            method_df = dataset_df[
                (dataset_df["model_variant"] == "deterministic")
                & (dataset_df["method"] == method)
                & (dataset_df["sample_count"] == primary_sample_count(method, cfg))
            ]
            merged = det.merge(method_df, on=["dataset", "seed"], suffixes=("_det", "_method"))
            if len(merged) < 2:
                continue
            for metric in metrics_to_test:
                base = merged[f"{metric}_det"].to_numpy()
                comp = merged[f"{metric}_method"].to_numpy()
                diff = comp - base
                try:
                    t_stat, p_val = stats.ttest_rel(comp, base)
                    t_stat = float(t_stat)
                    p_val = float(p_val)
                except Exception:
                    t_stat = float("nan")
                    p_val = float("nan")
                try:
                    w_stat, w_p = stats.wilcoxon(comp, base, zero_method="zsplit")
                    w_stat = float(w_stat)
                    w_p = float(w_p)
                except Exception:
                    w_stat = float("nan")
                    w_p = float("nan")
                ci_low, ci_high = mean_ci(diff)
                primary_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{method}_S{primary_sample_count(method, cfg)} vs deterministic",
                        "metric": metric,
                        "n_pairs": int(len(merged)),
                        "baseline_mean": float(base.mean()),
                        "comparison_mean": float(comp.mean()),
                        "mean_difference": float(diff.mean()),
                        "diff_ci_low": ci_low,
                        "diff_ci_high": ci_high,
                        "cohens_dz": cohens_dz(diff),
                        "paired_t": t_stat,
                        "paired_t_p": p_val,
                        "wilcoxon_w": w_stat,
                        "wilcoxon_p": w_p,
                    }
                )

    tests = pd.DataFrame(primary_rows)
    if tests.empty:
        return tests
    corrected = []
    for (_, metric), group in tests.groupby(["dataset", "metric"], sort=False):
        corrected.extend(zip(group.index, holm_adjust(group["paired_t_p"].fillna(1.0).tolist())))
    tests["paired_t_p_holm"] = np.nan
    for idx, adj in corrected:
        tests.loc[idx, "paired_t_p_holm"] = adj
    return tests.sort_values(["dataset", "metric", "comparison"])


def plot_metric_bars(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    primary = summary.copy()
    primary = primary[
        primary.apply(
            lambda row: row["sample_count"]
            == (1 if row["method"] in {"deterministic", "dropout_eval"} else 20),
            axis=1,
        )
    ]
    datasets = list(primary["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        df = primary[primary["dataset"] == dataset].copy()
        df["label"] = df["method"] + "\n(" + df["model_variant"] + ")"
        x = np.arange(len(df))
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        ax.bar(x, df[mean_col], yerr=df[std_col], capsize=4, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=35, ha="right")
        ax.set_title(dataset)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sample_sweep(metrics_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    methods = ["output_sample", "hidden_categorical", "sap", "gaussian", "mc_dropout"]
    datasets = list(metrics_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        df = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["method"].isin(methods))]
        grouped = df.groupby(["method", "sample_count"])[metric].agg(["mean", "std"]).reset_index()
        for method in methods:
            md = grouped[grouped["method"] == method]
            if md.empty:
                continue
            ax.errorbar(
                md["sample_count"],
                md["mean"],
                yerr=md["std"],
                marker="o",
                capsize=3,
                label=method,
            )
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Monte Carlo samples")
        ax.set_ylabel(metric)
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    datasets = list(history_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        df = history_df[history_df["dataset"] == dataset]
        grouped = df.groupby(["variant", "epoch"])["val_accuracy"].agg(["mean", "std"]).reset_index()
        for variant in grouped["variant"].unique():
            vd = grouped[grouped["variant"] == variant]
            ax.errorbar(vd["epoch"], vd["mean"], yerr=vd["std"], marker="o", capsize=3, label=variant)
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_environment(cfg: ExperimentConfig, device: torch.device) -> None:
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_mib": int(props.total_memory / 1024 / 1024),
                }
            )
    env = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_info": gpu_info,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "config": asdict(cfg),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (RESULTS_DIR / "environment.json").write_text(json.dumps(env, indent=2))


def run_experiments(cfg: ExperimentConfig) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = get_device()
    save_environment(cfg, device)

    all_metrics: list[dict] = []
    all_history: list[dict] = []
    all_data_metadata: list[dict] = []

    for dataset in cfg.datasets:
        for seed in cfg.seeds:
            set_seed(seed)
            train_loader, val_loader, test_loader, data_meta = prepare_data(dataset, seed, cfg)
            data_meta["seed"] = seed
            all_data_metadata.append(data_meta)

            trained_models: dict[str, SmallCNN] = {}
            for variant in ["deterministic", "dropout"]:
                model, history = train_model(
                    dataset, seed, variant, train_loader, val_loader, cfg, device
                )
                trained_models[variant] = model
                all_history.extend(history)
                pd.DataFrame(all_history).to_csv(RESULTS_DIR / "training_history.csv", index=False)

            for variant, method, sample_counts in method_grid():
                model = trained_models[variant]
                eval_method = "deterministic" if method == "dropout_eval" else method
                for sample_count in sample_counts:
                    if sample_count not in cfg.sample_counts and sample_count != 1:
                        continue
                    method_offsets = {
                        "deterministic": 11,
                        "output_sample": 23,
                        "hidden_categorical": 37,
                        "sap": 41,
                        "gaussian": 53,
                        "dropout_eval": 67,
                        "mc_dropout": 79,
                    }
                    set_seed(seed + sample_count + method_offsets[method])
                    metrics, _probs, _labels = evaluate_method(
                        model, test_loader, device, eval_method, sample_count, cfg
                    )
                    metrics.update(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model_variant": variant,
                            "method": method,
                        }
                    )
                    all_metrics.append(metrics)
                    pd.DataFrame(all_metrics).to_csv(RESULTS_DIR / "metrics_raw.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    history_df = pd.DataFrame(all_history)
    metadata_df = pd.DataFrame(all_data_metadata)

    metrics_df.to_csv(RESULTS_DIR / "metrics_raw.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    metadata_df.to_csv(RESULTS_DIR / "data_metadata.csv", index=False)
    (RESULTS_DIR / "metrics_raw.json").write_text(metrics_df.to_json(orient="records", indent=2))

    summary = summarize_metrics(metrics_df)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    tests = statistical_tests(metrics_df, cfg)
    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    plot_training_curves(history_df, FIGURES_DIR / "training_curves.png")
    plot_metric_bars(summary, "accuracy", FIGURES_DIR / "accuracy_primary.png")
    plot_metric_bars(summary, "ece", FIGURES_DIR / "ece_primary.png")
    plot_metric_bars(summary, "nll", FIGURES_DIR / "nll_primary.png")
    plot_sample_sweep(metrics_df, "accuracy", FIGURES_DIR / "accuracy_sample_sweep.png")
    plot_sample_sweep(metrics_df, "prob_variance", FIGURES_DIR / "variance_sample_sweep.png")
    plot_sample_sweep(metrics_df, "entropy", FIGURES_DIR / "entropy_sample_sweep.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test setup.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names under datasets/.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds.")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    updates = {}
    if args.quick:
        updates.update(
            {
                "datasets": ("mnist",),
                "seeds": (42,),
                "train_size": 2_000,
                "val_size": 500,
                "test_size": 1_000,
                "epochs": 2,
                "patience": 2,
                "sample_counts": (1, 5),
            }
        )
    if args.datasets is not None:
        updates["datasets"] = tuple(args.datasets)
    if args.seeds is not None:
        updates["seeds"] = tuple(args.seeds)
    for key in ["train_size", "val_size", "test_size", "epochs", "batch_size"]:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            updates[key] = value
    return replace(cfg, **updates)


def main() -> None:
    cfg = config_from_args(parse_args())
    run_experiments(cfg)


if __name__ == "__main__":
    main()
