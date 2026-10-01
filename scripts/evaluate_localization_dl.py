#!/usr/bin/env python3
"""Run Stage 2 (OpenCV localization) + Stage 3 (CNN) on a whiteboard photo.

Draws predicted boxes/labels and writes ``data/evaluation_output.png``.

If ``--image`` is omitted, uses files in ``data/test_real/``. If that folder
is empty, builds a synthetic multi-shape board so the pipeline can be checked
before real photos exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.localization import localize  # noqa: E402
from src.model import (  # noqa: E402
    CLASS_NAMES,
    get_eval_transforms,
    load_checkpoint,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "shape_classifier.pth"
DEFAULT_REAL_DIR = PROJECT_ROOT / "data" / "test_real"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation_output.png"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# BGR colors for overlay (box, circle, diamond, arrow).
LABEL_COLORS = {
    "box": (40, 180, 40),
    "circle": (200, 120, 20),
    "diamond": (180, 40, 180),
    "arrow": (30, 90, 220),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize candidates and classify them with MobileNetV2.",
    )
    parser.add_argument("--image", type=Path, default=None, help="Single input photo.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_REAL_DIR,
        help="Folder of real test photos (used when --image is omitted).",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Skip drawing predictions below this softmax score (0 keeps all).",
    )
    return parser.parse_args(argv)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return files


def build_demo_whiteboard(seed: int = 7) -> np.ndarray:
    """Compose a few synthetic shapes onto one board (worked-example layout)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from generate_synthetic_data import render_shape  # local script import

    rng = np.random.default_rng(seed)
    board = np.full((720, 1100, 3), 240, dtype=np.uint8)
    # Subtle gradient so localization is not on a perfectly flat field.
    yy = np.linspace(0, 12, board.shape[0], dtype=np.float32)
    board = np.clip(board.astype(np.int16) + yy[:, None, None].astype(np.int16), 0, 255).astype(np.uint8)

    layout = [
        ("box", (40, 80)),
        ("box", (430, 80)),
        ("box", (820, 80)),
        ("arrow", (250, 280)),
        ("arrow", (640, 280)),
        ("circle", (80, 430)),
        ("diamond", (470, 430)),
        ("arrow", (300, 500)),
    ]
    for class_name, (x, y) in layout:
        crop = render_shape(class_name, rng)
        h, w = crop.shape[:2]
        y1, x1 = min(y + h, board.shape[0]), min(x + w, board.shape[1])
        patch = crop[: y1 - y, : x1 - x]
        # Paste darker ink over the board (min keeps marker pixels).
        roi = board[y:y1, x:x1]
        board[y:y1, x:x1] = np.minimum(roi, patch)
    return board


def resolve_input_image(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return image, str(args.image)

    real = list_images(args.input_dir)
    if real:
        path = real[0]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if len(real) > 1:
            print(f"Found {len(real)} images in {args.input_dir}; using {path.name}")
        return image, str(path)

    print(
        f"No photos in {args.input_dir}. Building a synthetic demo whiteboard "
        "so localization + classification can still be checked."
    )
    return build_demo_whiteboard(), "synthetic-demo"


def crop_to_tensor(crop_bgr: np.ndarray, tfm, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = tfm(pil).unsqueeze(0).to(device)
    return tensor


def classify_crops(
    crops: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
) -> list[tuple[str, float]]:
    if not crops:
        return []
    tfm = get_eval_transforms()
    model.eval()
    results: list[tuple[str, float]] = []
    with torch.no_grad():
        for crop in crops:
            logits = model(crop_to_tensor(crop, tfm, device))
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            results.append((CLASS_NAMES[idx], float(probs[idx].item())))
    return results


def draw_predictions(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    kinds: list[str],
    preds: list[tuple[str, float]],
    min_confidence: float,
) -> np.ndarray:
    vis = image.copy()
    for bbox, kind, (label, conf) in zip(boxes, kinds, preds):
        if conf < min_confidence:
            continue
        x, y, w, h = bbox
        color = LABEL_COLORS.get(label, (0, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        caption = f"{label} {conf:.2f} [{kind}]"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y - 6)
        cv2.rectangle(vis, (x, y_text - th - 4), (x + tw + 4, y_text + 2), color, -1)
        cv2.putText(
            vis,
            caption,
            (x + 2, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.weights.is_file():
        print(
            f"Missing weights at {args.weights}\n"
            "Train first: python scripts/train_classifier.py",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")

    model = load_checkpoint(args.weights, device=device)
    image, source = resolve_input_image(args)
    print(f"Input: {source}  shape={image.shape[1]}x{image.shape[0]}")

    loc = localize(image)
    candidates = loc.all_candidates
    print(
        f"Localized {len(loc.shapes)} shape candidate(s), "
        f"{len(loc.arrows)} arrow candidate(s)."
    )

    preds = classify_crops([c.crop for c in candidates], model, device)
    vis = draw_predictions(
        image,
        [c.bbox for c in candidates],
        [c.kind for c in candidates],
        preds,
        args.min_confidence,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), vis):
        print(f"Failed to write {args.output}", file=sys.stderr)
        return 1

    print("Predictions:")
    for cand, (label, conf) in zip(candidates, preds):
        print(
            f"  {cand.kind:6s} bbox={cand.bbox} -> {label:8s}  conf={conf:.3f}"
        )
    print(f"Wrote visualization: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
