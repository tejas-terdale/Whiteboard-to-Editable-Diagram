#!/usr/bin/env python3
"""Run Hough connector matching on the clean flowchart fixtures (no OCR)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DEFAULT_WEIGHTS, build_labeled_demo_whiteboard, load_classifier, run_pipeline

DEFAULT_DEBUG = PROJECT_ROOT / "data" / "debug"
FIXTURES = [
    PROJECT_ROOT / "data" / "test_real" / "flowchart_clean_a.jpg",
    PROJECT_ROOT / "data" / "test_real" / "flowchart_clean_b.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arrow/Hough relationship test.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG)
    return parser.parse_args()


def run_one(image, name: str, model, device, debug_dir: Path) -> None:
    out_dir = debug_dir / name
    result = run_pipeline(
        image,
        model=model,
        device=device,
        skip_ocr=True,
        debug_dir=out_dir,
    )
    print(f"\n=== {name} ===")
    for line in result.log_lines:
        if line.startswith("  pairing"):
            continue
        print(" ", line)
    print(
        f" nodes={result.n_nodes}  edges={result.n_edges}  "
        f"contour_arrows={result.n_arrows_classified}  hough={result.n_hough_lines}"
    )
    print(f" debug={result.debug_image_path}")
    g = result.graph
    print(" edges:")
    if g.number_of_edges() == 0:
        print("   (none)")
    for u, v in sorted(tuple(sorted(e)) for e in g.edges()):
        la = g.nodes[u].get("label", "")
        lb = g.nodes[v].get("label", "")
        print(
            f"   {u}:{g.nodes[u].get('shape_type')}({la!r}) -- "
            f"{v}:{g.nodes[v].get('shape_type')}({lb!r})"
        )


def main() -> int:
    args = parse_args()
    if not DEFAULT_WEIGHTS.is_file():
        print(f"Missing {DEFAULT_WEIGHTS}", file=sys.stderr)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_classifier(DEFAULT_WEIGHTS, device)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read {args.image}", file=sys.stderr)
            return 1
        run_one(image, args.image.stem, model, device, args.debug_dir)
        return 0

    run_one(build_labeled_demo_whiteboard(), "synthetic_demo", model, device, args.debug_dir)
    for path in FIXTURES:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        run_one(image, path.stem, model, device, args.debug_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
