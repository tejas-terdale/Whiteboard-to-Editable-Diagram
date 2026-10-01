#!/usr/bin/env python3
"""Generate synthetic whiteboard-style crops for shape/arrow classification.

Produces 224x224 PNG images for four classes (box, circle, diamond, arrow)
with hand-drawn jitter, randomized stroke, rotation/scale, and camera-like
background noise. Labels are implicit in the output folder names.

Example:
    python scripts/generate_synthetic_data.py --num-train 1000 --num-val 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

CLASSES = ("box", "circle", "diamond", "arrow")
IMAGE_SIZE = 224
RNG_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

# Margin so rotated strokes stay inside the crop.
MARGIN = 28


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate 2D points around their centroid."""
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    center = points.mean(axis=0)
    return (points - center) @ rot.T + center


def _resample_polyline(
    vertices: np.ndarray,
    samples_per_edge: int,
    closed: bool,
) -> np.ndarray:
    """Densify a polyline so jitter can wobble along each edge."""
    if closed:
        verts = np.vstack([vertices, vertices[0]])
    else:
        verts = vertices

    pieces: list[np.ndarray] = []
    for i in range(len(verts) - 1):
        t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        segment = (1.0 - t)[:, None] * verts[i] + t[:, None] * verts[i + 1]
        pieces.append(segment)
    if not closed:
        pieces.append(verts[-1][None, :])
    return np.vstack(pieces)


def _smooth_closed(noise: np.ndarray, kernel_size: int) -> np.ndarray:
    """Moving-average smooth periodic (closed-curve) 2D noise."""
    k = max(3, kernel_size | 1)  # odd
    pad = k // 2
    padded = np.concatenate([noise[-pad:], noise, noise[:pad]], axis=0)
    kernel = np.ones(k, dtype=np.float64) / k
    smoothed = np.zeros_like(noise)
    for dim in range(2):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def _smooth_open(noise: np.ndarray, kernel_size: int) -> np.ndarray:
    """Moving-average smooth open-curve 2D noise (replicate padding)."""
    k = max(3, kernel_size | 1)
    pad = k // 2
    padded = np.pad(noise, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    smoothed = np.zeros_like(noise)
    for dim in range(2):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def organic_jitter(
    points: np.ndarray,
    rng: np.random.Generator,
    closed: bool,
    amp_min: float = 2.0,
    amp_max: float = 6.0,
) -> np.ndarray:
    """Perturb vertices with spatially-smoothed noise (hand-drawn wobble)."""
    amplitude = float(rng.uniform(amp_min, amp_max))
    raw = rng.normal(0.0, 1.0, size=points.shape)
    kernel = int(rng.integers(5, 13))
    if closed:
        smoothed = _smooth_closed(raw, kernel)
    else:
        smoothed = _smooth_open(raw, kernel)
    # Normalize so typical displacement is ~amplitude pixels.
    rms = np.sqrt(np.mean(smoothed**2)) + 1e-6
    jittered = points + smoothed * (amplitude / rms)
    return jittered


def _fit_in_canvas(
    points: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scale/translate points so they sit in the crop with a random offset."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = np.maximum(maxs - mins, 1.0)

    usable = IMAGE_SIZE - 2 * MARGIN
    scale = usable / float(np.max(size))
    # Extra random shrink so shapes are not all the same size.
    scale *= float(rng.uniform(0.55, 0.95))
    centered = (points - (mins + maxs) / 2.0) * scale
    jitter_xy = rng.uniform(-10.0, 10.0, size=2)
    return centered + IMAGE_SIZE / 2.0 + jitter_xy


# ---------------------------------------------------------------------------
# Shape constructors (canonical geometry, then transform + jitter)
# ---------------------------------------------------------------------------


def _aspect_size(rng: np.random.Generator) -> tuple[float, float]:
    """Random width/height used before rotation."""
    base = float(rng.uniform(90.0, 150.0))
    aspect = float(rng.uniform(0.65, 1.45))
    width = base * np.sqrt(aspect)
    height = base / np.sqrt(aspect)
    return width, height


def make_box_points(rng: np.random.Generator) -> np.ndarray:
    w, h = _aspect_size(rng)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
        dtype=np.float64,
    )
    dense = _resample_polyline(corners, samples_per_edge=18, closed=True)
    return dense


def make_diamond_points(rng: np.random.Generator) -> np.ndarray:
    w, h = _aspect_size(rng)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array(
        [[0.0, -hh], [hw, 0.0], [0.0, hh], [-hw, 0.0]],
        dtype=np.float64,
    )
    dense = _resample_polyline(corners, samples_per_edge=18, closed=True)
    return dense


def make_circle_points(rng: np.random.Generator) -> np.ndarray:
    """Ellipse sampled densely, then treated like a closed contour."""
    w, h = _aspect_size(rng)
    rx, ry = w / 2.0, h / 2.0
    n = int(rng.integers(48, 72))
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([rx * np.cos(theta), ry * np.sin(theta)], axis=1)


def make_arrow_points(rng: np.random.Generator) -> np.ndarray:
    """Open polyline: shaft plus a V-shaped head (whiteboard-style arrow)."""
    length = float(rng.uniform(100.0, 160.0))
    head_len = float(rng.uniform(18.0, 32.0))
    head_w = float(rng.uniform(10.0, 20.0))
    # Slight bow so not every arrow is perfectly straight.
    bow = float(rng.uniform(-18.0, 18.0))

    n_shaft = 20
    t = np.linspace(0.0, 1.0, n_shaft)
    shaft_x = t * length - length / 2.0
    shaft_y = bow * np.sin(np.pi * t)
    shaft = np.stack([shaft_x, shaft_y], axis=1)

    tip = shaft[-1]
    # Tangent near the tip for a stable head orientation.
    tangent = shaft[-1] - shaft[-3]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
    normal = np.array([-tangent[1], tangent[0]])
    left = tip - tangent * head_len + normal * head_w
    right = tip - tangent * head_len - normal * head_w

    # Draw as one stroke: along shaft, then head as a V returning through tip.
    head = np.vstack([left, tip, right])
    return np.vstack([shaft, head])


SHAPE_BUILDERS: dict[str, Callable[[np.random.Generator], np.ndarray]] = {
    "box": make_box_points,
    "circle": make_circle_points,
    "diamond": make_diamond_points,
    "arrow": make_arrow_points,
}


# ---------------------------------------------------------------------------
# Rendering (whiteboard look)
# ---------------------------------------------------------------------------


def make_whiteboard_background(rng: np.random.Generator) -> np.ndarray:
    """Off-white canvas with a faint uneven grayscale gradient."""
    base = int(rng.integers(232, 248))
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), base, dtype=np.uint8)

    # Slow bilinear gradient (simulates uneven lighting / camera vignetting).
    corners = rng.integers(-18, 19, size=(2, 2))
    ys = np.linspace(0.0, 1.0, IMAGE_SIZE)
    xs = np.linspace(0.0, 1.0, IMAGE_SIZE)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    gradient = (
        corners[0, 0] * (1 - grid_x) * (1 - grid_y)
        + corners[0, 1] * grid_x * (1 - grid_y)
        + corners[1, 0] * (1 - grid_x) * grid_y
        + corners[1, 1] * grid_x * grid_y
    )
    canvas = np.clip(canvas.astype(np.int16) + gradient[:, :, None], 0, 255)
    return canvas.astype(np.uint8)


def add_camera_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise, occasional salt-and-pepper, light blur."""
    noisy = image.astype(np.float32)
    sigma = float(rng.uniform(2.0, 7.0))
    noisy += rng.normal(0.0, sigma, size=noisy.shape)

    # Salt-and-pepper on a small fraction of pixels.
    amount = float(rng.uniform(0.0005, 0.003))
    n_pixels = int(amount * IMAGE_SIZE * IMAGE_SIZE)
    if n_pixels > 0:
        ys = rng.integers(0, IMAGE_SIZE, size=n_pixels)
        xs = rng.integers(0, IMAGE_SIZE, size=n_pixels)
        salt = rng.random(n_pixels) > 0.5
        noisy[ys[salt], xs[salt]] = 255
        noisy[ys[~salt], xs[~salt]] = rng.integers(40, 90)

    out = np.clip(noisy, 0, 255).astype(np.uint8)
    ksize = int(rng.choice([0, 3, 3, 5]))  # often a little blur, sometimes none
    if ksize >= 3:
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    return out


def stroke_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Dark gray through pure black (BGR)."""
    shade = int(rng.integers(0, 55))
    return (shade, shade, shade)


def render_shape(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Build one 224x224 whiteboard-style crop for `class_name`."""
    canvas = make_whiteboard_background(rng)
    points = SHAPE_BUILDERS[class_name](rng)

    angle = float(rng.uniform(-15.0, 15.0))
    points = _rotate(points, angle)
    points = _fit_in_canvas(points, rng)

    closed = class_name != "arrow"
    points = organic_jitter(points, rng, closed=closed)

    thickness = int(rng.integers(2, 6))  # 2..5 inclusive
    color = stroke_color(rng)
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    line_type = cv2.LINE_AA
    if closed:
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=line_type)
    else:
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=line_type)

    return add_camera_noise(canvas, rng)


# ---------------------------------------------------------------------------
# I/O / CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 224x224 whiteboard crops for 4 classes.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Images per class in the training split (default: 1000).",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=200,
        help="Images per class in the validation split (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder for train/val class subdirectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help="Base RNG seed (default: 42).",
    )
    return parser.parse_args(argv)


def _progress(done: int, total: int, label: str, width: int = 28) -> None:
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r  [{bar}] {done}/{total} {label}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def generate_split(
    split: str,
    num_per_class: int,
    output_dir: Path,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Write `num_per_class` images for each class into output_dir/split/class."""
    counts: dict[str, int] = {}
    for class_name in CLASSES:
        dest = output_dir / split / class_name
        dest.mkdir(parents=True, exist_ok=True)
        total = num_per_class
        for i in range(total):
            image = render_shape(class_name, rng)
            path = dest / f"{class_name}_{split}_{i:05d}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Failed to write {path}")
            if (i + 1) % 25 == 0 or i + 1 == total:
                _progress(i + 1, total, f"{split}/{class_name}")
        counts[class_name] = total
    return counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_train < 0 or args.num_val < 0:
        print("Error: --num-train and --num-val must be >= 0", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    print("Synthetic whiteboard dataset generator")
    print(f"  output : {output_dir}")
    print(f"  classes: {', '.join(CLASSES)}")
    print(f"  size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  train  : {args.num_train} / class")
    print(f"  val    : {args.num_val} / class")
    print(f"  seed   : {args.seed}")
    print()

    train_counts = generate_split("train", args.num_train, output_dir, rng)
    val_counts = generate_split("val", args.num_val, output_dir, rng)

    elapsed = time.perf_counter() - started
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())

    print()
    print("Completed.")
    print(f"  train images : {train_total} ({args.num_train} x {len(CLASSES)} classes)")
    print(f"  val images   : {val_total} ({args.num_val} x {len(CLASSES)} classes)")
    print(f"  grand total  : {train_total + val_total}")
    print(f"  elapsed      : {elapsed:.1f}s")
    print(f"  layout       : {output_dir}/{{train,val}}/{{{','.join(CLASSES)}}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
