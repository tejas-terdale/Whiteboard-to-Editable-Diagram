"""MobileNetV2 transfer-learning classifier for whiteboard shape crops.

Classes: box, circle, diamond, arrow. Crops are 224x224, normalized with
ImageNet mean/std so the pretrained backbone stays in-distribution.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights

CLASS_NAMES: tuple[str, ...] = ("box", "circle", "diamond", "arrow")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# How many trailing MobileNetV2 feature blocks stay trainable.
UNFREEZE_LAST_N = 2


class PadToSquare:
    """Letterbox a PIL image so Resize(224,224) does not stretch thin arrows."""

    def __init__(self, fill: int = 240) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        side = max(width, height)
        if side == width and side == height:
            return img
        color = (self.fill, self.fill, self.fill) if img.mode == "RGB" else self.fill
        canvas = Image.new(img.mode, (side, side), color=color)
        canvas.paste(img, ((side - width) // 2, (side - height) // 2))
        return canvas


def get_train_transforms() -> transforms.Compose:
    """Augment synthetic crops so the model is less locked to perfect renders."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            PadToSquare(fill=240),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = True,
    freeze_early: bool = True,
    unfreeze_last_n: int = UNFREEZE_LAST_N,
) -> nn.Module:
    """Pretrained MobileNetV2 with a 4-way linear head.

    Early ``features`` blocks stay frozen; the last ``unfreeze_last_n`` blocks
    and the new classifier head are trainable (Stage 3 spec).
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_early:
        for param in model.features.parameters():
            param.requires_grad = False
        n_blocks = len(model.features)
        start = max(0, n_blocks - unfreeze_last_n)
        for block in model.features[start:]:
            for param in block.parameters():
                param.requires_grad = True

    return model


def load_imagefolder(
    root: Path,
    train: bool,
) -> datasets.ImageFolder:
    """Folder layout: ``root/{class_name}/*.png`` (ImageFolder convention)."""
    tfm = get_train_transforms() if train else get_eval_transforms()
    dataset = datasets.ImageFolder(str(root), transform=tfm)
    if tuple(dataset.classes) != CLASS_NAMES:
        # ImageFolder sorts class folder names alphabetically.
        # Ours already sort as arrow, box, circle, diamond — we remap via class_to_idx.
        missing = set(CLASS_NAMES) - set(dataset.classes)
        if missing:
            raise ValueError(f"{root} is missing class folders: {sorted(missing)}")
    return dataset


def class_to_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(CLASS_NAMES)}


def index_to_class() -> dict[int, str]:
    return {i: name for i, name in enumerate(CLASS_NAMES)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": list(CLASS_NAMES),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def classify_bgr_crops(
    crops: list,
    model: nn.Module,
    device: torch.device,
) -> list[tuple[str, float]]:
    """Run the classifier on OpenCV BGR crops; returns ``(class_name, confidence)``."""
    import cv2
    import numpy as np

    if not crops:
        return []
    tfm = get_eval_transforms()
    model.eval()
    results: list[tuple[str, float]] = []
    with torch.no_grad():
        for crop in crops:
            arr = np.asarray(crop)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            tensor = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0]
            idx = int(torch.argmax(probs).item())
            results.append((CLASS_NAMES[idx], float(probs[idx].item())))
    return results


def load_checkpoint(
    path: Path,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    """Load weights saved by ``save_checkpoint`` (or a raw state_dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=pretrained, freeze_early=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
