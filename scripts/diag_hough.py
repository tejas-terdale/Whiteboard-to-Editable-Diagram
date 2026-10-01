#!/usr/bin/env python3
"""Localization + Hough matching diagnostic (no CNN/OCR)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.localization import localize, preprocess, render_line_debug
from src.relationships import (
    ShapeRef,
    collect_arrow_refs,
    match_arrows_to_shapes,
    render_relationship_debug,
    MatchDebug,
    ArrowRef,
)


def run(path: Path, out_dir: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    print(f"\n=== {path.name}  {image.shape[1]}x{image.shape[0]} ===")
    loc = localize(image)
    print(f" shapes={len(loc.shapes)} contour_arrows={len(loc.arrows)} "
          f"raw_hough={len(loc.raw_lines)} merged_lines={len(loc.lines)}")
    for i, s in enumerate(loc.shapes):
        x, y, w, h = s.bbox
        print(f"  shape {i} bbox=({x},{y},{w},{h}) aspect={s.metrics.get('aspect_bbox',0):.2f} "
              f"sol={s.metrics.get('solidity',0):.2f}")
    for i, a in enumerate(loc.arrows[:12]):
        x, y, w, h = a.bbox
        print(f"  contour_arrow {i} bbox=({x},{y},{w},{h}) len~{max(w,h)}")
    for i, ln in enumerate(loc.lines):
        print(f"  line {i} {ln.source} {ln.p1} -> {ln.p2} len={ln.length:.1f}")

    shapes = [ShapeRef(node_id=i, bbox=c.bbox, contour=c.contour) for i, c in enumerate(loc.shapes)]
    arrows = collect_arrow_refs(
        [ArrowRef(contour=a.contour, bbox=a.bbox) for a in loc.arrows],
        loc.lines,
    )
    dbg = MatchDebug()
    h, w = image.shape[:2]
    matches = match_arrows_to_shapes(arrows, shapes, image_size=(h, w), debug=dbg, verbose=True)
    print(f" matches={len(matches)} rejected={len(dbg.rejected)} maxd={dbg.max_endpoint_distance:.1f}")
    for m in matches:
        print(f"  EDGE {m.node_a}--{m.node_b} d={m.dist_a:.1f}/{m.dist_b:.1f} arrow={m.arrow_index}")
    for r in dbg.rejected:
        print(f"  REJ {r.reason} nodes={r.node_a},{r.node_b} d={r.dist_a:.1f}/{r.dist_b:.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    render_line_debug(image, loc, path=out_dir / f"{path.stem}_lines.png")
    vis = render_relationship_debug(
        image, shapes, loc.raw_lines, loc.lines, matches, debug=dbg,
        path=out_dir / f"{path.stem}_rel.png",
    )
    # also dump binary + ink mask
    gray, blurred, binary = preprocess(image)
    cv2.imwrite(str(out_dir / f"{path.stem}_binary.png"), binary)
    print(f" wrote {out_dir / (path.stem + '_rel.png')}")
    print(f" wrote {out_dir / (path.stem + '_lines.png')}")
    print(
        f" HoughLinesP threshold={loc.hough_params.threshold} "
        f"minLineLength={loc.hough_params.min_line_length} "
        f"maxLineGap={loc.hough_params.max_line_gap}"
    )


def main() -> int:
    out = PROJECT_ROOT / "data" / "debug"
    files = [
        PROJECT_ROOT / "data" / "test_real" / "flowchart_clean_a.jpg",
        PROJECT_ROOT / "data" / "test_real" / "flowchart_clean_b.jpg",
        PROJECT_ROOT / "data" / "test_real" / "Gemini_Generated_Image_iwbazkiwbazkiwba.png",
        PROJECT_ROOT / "data" / "test_real" / "Gemini_Generated_Image_jff3kjjff3kjjff3.png",
    ]
    for p in files:
        run(p, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
