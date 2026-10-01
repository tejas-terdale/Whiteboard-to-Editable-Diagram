#!/usr/bin/env python3
"""Train the MobileNetV2 shape/arrow classifier on synthetic crops.

Reads ``data/synthetic/train`` and ``data/synthetic/val`` (ImageFolder layout)
and writes the best validation-accuracy checkpoint to
``models/shape_classifier.pth``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (  # noqa: E402
    CLASS_NAMES,
    build_model,
    load_imagefolder,
    save_checkpoint,
)

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "synthetic" / "train"
DEFAULT_VAL = PROJECT_ROOT / "data" / "synthetic" / "val"
DEFAULT_OUT = PROJECT_ROOT / "models" / "shape_classifier.pth"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 shape classifier.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the new head.")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="LR for unfrozen backbone blocks.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model: nn.Module, lr: float, backbone_lr: float) -> torch.optim.Optimizer:
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")]
    groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.Adam(groups)


def confusion_and_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, dict[str, tuple[float, float, int]]]:
    """Per-class precision/recall plus a confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    scores: dict[str, tuple[float, float, int]] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        scores[name] = (precision, recall, support)
    return cm, scores


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_seen = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        n_seen += batch
        preds = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    acc = float((y_true == y_pred).mean()) if n_seen else 0.0
    return total_loss / max(n_seen, 1), acc, y_true, y_pred


def log_scores(split: str, loss: float, acc: float, scores: dict[str, tuple[float, float, int]]) -> None:
    print(f"  {split:5s}  loss={loss:.4f}  acc={acc:.3f}")
    for name in CLASS_NAMES:
        precision, recall, support = scores[name]
        print(
            f"         {name:8s}  P={precision:.3f}  R={recall:.3f}  n={support}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    if not args.train_dir.is_dir() or not args.val_dir.is_dir():
        print(
            "Missing synthetic data. Generate it first:\n"
            "  python scripts/generate_synthetic_data.py",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")

    train_ds = load_imagefolder(args.train_dir, train=True)
    val_ds = load_imagefolder(args.val_dir, train=False)
    print(f"Train images: {len(train_ds)}  classes={train_ds.classes}")
    print(f"Val images:   {len(val_ds)}  classes={val_ds.classes}")

    # ImageFolder indexes by sorted folder names; remap labels to CLASS_NAMES order.
    name_to_model_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    folder_idx_to_model = {
        folder_idx: name_to_model_idx[name]
        for name, folder_idx in train_ds.class_to_idx.items()
    }

    def remap_collate(batch: list[tuple[torch.Tensor, int]]):
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor(
            [folder_idx_to_model[item[1]] for item in batch],
            dtype=torch.long,
        )
        return images, labels

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=remap_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=remap_collate,
    )

    model = build_model(pretrained=True, freeze_early=True).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.lr, args.backbone_lr)

    best_acc = -1.0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, y_tr, p_tr = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        _, train_scores = confusion_and_scores(y_tr, p_tr, len(CLASS_NAMES))
        log_scores("train", train_loss, train_acc, train_scores)

        val_loss, val_acc, y_va, p_va = run_epoch(
            model, val_loader, criterion, None, device
        )
        _, val_scores = confusion_and_scores(y_va, p_va, len(CLASS_NAMES))
        log_scores("val", val_loss, val_acc, val_scores)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                args.output,
                model,
                extra={
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_to_idx": name_to_model_idx,
                },
            )
            print(f"  saved best checkpoint -> {args.output}  (val acc={best_acc:.3f})")

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s. Best val acc={best_acc:.3f}")
    print(f"Checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
