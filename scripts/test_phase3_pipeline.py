#!/usr/bin/env python3
"""End-to-end Phase 3 test: localize → classify → OCR → match edges → NetworkX.

Writes a JSON snapshot to ``data/phase3_output.json``.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import build_graph, save_graph_json  # noqa: E402
from src.localization import localize  # noqa: E402
from src.model import CLASS_NAMES, classify_bgr_crops, load_checkpoint  # noqa: E402
from src.ocr import SHAPE_CLASSES, ShapeOCR  # noqa: E402
from src.relationships import (  # noqa: E402
    ArrowRef,
    MatchDebug,
    ShapeRef,
    collect_arrow_refs,
    match_arrows_to_shapes,
    render_relationship_debug,
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "shape_classifier.pth"
DEFAULT_REAL_DIR = PROJECT_ROOT / "data" / "test_real"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "phase3_output.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 pipeline integration test.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=0.25,
        help="EasyOCR detections below this score are ignored.",
    )
    parser.add_argument(
        "--ocr-gpu",
        action="store_true",
        help="Run EasyOCR on CUDA (off by default; more stable on CPU).",
    )
    return parser.parse_args(argv)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def build_labeled_demo_whiteboard() -> np.ndarray:
    """Worked-example board: Start — Process Data — End, plus a circle/diamond."""
    board = np.full((640, 1100, 3), 242, dtype=np.uint8)
    yy = np.linspace(0, 10, board.shape[0], dtype=np.float32)
    board = np.clip(board.astype(np.int16) + yy[:, None, None].astype(np.int16), 0, 255)
    board = board.astype(np.uint8)

    ink = (25, 25, 25)
    # Keep a visible gap between arrows and shapes so strokes stay separate contours.
    boxes = [
        ((60, 170, 200, 130), "Start"),
        ((430, 170, 250, 130), "Process Data"),
        ((820, 170, 200, 130), "End"),
    ]
    for (x, y, w, h), text in boxes:
        cv2.rectangle(board, (x, y), (x + w, y + h), ink, 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(board, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ink, 2, cv2.LINE_AA)

    def _arrow(p0: tuple[int, int], p1: tuple[int, int]) -> None:
        cv2.line(board, p0, p1, ink, 3, cv2.LINE_AA)
        vec = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float64)
        norm = np.linalg.norm(vec) + 1e-6
        unit = vec / norm
        left = np.array([-unit[1], unit[0]])
        tip = np.array(p1, dtype=np.float64)
        a = (tip - unit * 16 + left * 9).astype(int)
        b = (tip - unit * 16 - left * 9).astype(int)
        cv2.line(board, p1, tuple(a), ink, 3, cv2.LINE_AA)
        cv2.line(board, p1, tuple(b), ink, 3, cv2.LINE_AA)

    _arrow((280, 235), (410, 235))
    _arrow((700, 235), (800, 235))

    cv2.circle(board, (160, 490), 55, ink, 3)
    cv2.putText(board, "Note", (128, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 2, cv2.LINE_AA)

    diamond = np.array([[720, 430], [800, 490], [720, 550], [640, 490]], dtype=np.int32)
    cv2.polylines(board, [diamond], isClosed=True, color=ink, thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(board, "Wait", (688, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 2, cv2.LINE_AA)
    _arrow((235, 470), (620, 490))

    return board


def _bbox_contained(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
    margin: int = 6,
) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return (
        ix >= ox - margin
        and iy >= oy - margin
        and ix + iw <= ox + ow + margin
        and iy + ih <= oy + oh + margin
    )


def drop_nested_shape_nodes(nodes: list[dict]) -> list[dict]:
    """Drop crops that sit inside a larger shape (usually interior text)."""
    survivors: list[dict] = []
    for i, node in enumerate(nodes):
        nested = False
        for j, other in enumerate(nodes):
            if i == j:
                continue
            if _bbox_contained(node["bbox"], other["bbox"]):
                nested = True
                break
        if not nested:
            survivors.append(node)
    for new_id, node in enumerate(survivors):
        node["id"] = new_id
    return survivors


def resolve_image(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {args.image}")
        return image, str(args.image)

    real = list_images(args.input_dir)
    if real:
        image = cv2.imread(str(real[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {real[0]}")
        return image, str(real[0])

    print(f"No photos in {args.input_dir}; using a labeled synthetic demo board.")
    return build_labeled_demo_whiteboard(), "synthetic-labeled-demo"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.weights.is_file():
        print(f"Missing {args.weights}. Train with scripts/train_classifier.py", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_checkpoint(args.weights, device=device)
    image, source = resolve_image(args)
    h, w = image.shape[:2]
    print(f"Input: {source}  ({w}x{h})")

    # A. Localization
    loc = localize(image)
    candidates = loc.all_candidates
    print(f"A. Localized {len(loc.shapes)} compact + {len(loc.arrows)} elongated contour(s).")

    # B. CNN classification (source of truth for shape vs arrow)
    preds = classify_bgr_crops([c.crop for c in candidates], model, device)
    shape_nodes: list[dict] = []
    arrow_refs: list[ArrowRef] = []
    node_id = 0
    for cand, (label, conf) in zip(candidates, preds):
        if label in SHAPE_CLASSES:
            shape_nodes.append(
                {
                    "id": node_id,
                    "shape_type": label,
                    "bbox": cand.bbox,
                    "contour": cand.contour,
                    "crop": cand.crop,
                    "confidence": conf,
                    "label": f"Node_{node_id}",
                }
            )
            node_id += 1
        elif label == "arrow":
            arrow_refs.append(ArrowRef(contour=cand.contour, bbox=cand.bbox))
        else:
            print(f"  skip unknown class {label}")

    n_contour_arrows = len(arrow_refs)
    arrow_refs = collect_arrow_refs(arrow_refs, loc.lines)

    before = len(shape_nodes)
    shape_nodes = drop_nested_shape_nodes(shape_nodes)
    if len(shape_nodes) != before:
        print(f"   dropped {before - len(shape_nodes)} nested crop(s) (likely interior text).")

    print(
        f"B. Classifier: {len(shape_nodes)} shape node(s), {n_contour_arrows} contour arrow(s) "
        f"+ {len(loc.lines)} Hough connector(s) (classes={list(CLASS_NAMES)})."
    )

    # C. OCR on shape crops only
    ocr = ShapeOCR(min_confidence=args.ocr_min_confidence, gpu=args.ocr_gpu)
    for node in shape_nodes:
        result = ocr.read_shape(node["crop"], node["id"])
        node["label"] = result.text
        node["ocr_confidence"] = result.confidence
        node["ocr_fallback"] = result.used_fallback
        print(
            f"C. node {node['id']} ({node['shape_type']}) "
            f"-> '{result.text}'  ocr_conf={result.confidence:.2f}"
            f"{' [fallback]' if result.used_fallback else ''}"
        )

    # D. Geometric undirected edges (Hough lines + contour arrows)
    shape_refs = [
        ShapeRef(node_id=n["id"], bbox=n["bbox"], contour=n["contour"]) for n in shape_nodes
    ]
    match_debug = MatchDebug()
    matches = match_arrows_to_shapes(
        arrow_refs,
        shape_refs,
        max_endpoint_distance=None,
        image_size=(h, w),
        debug=match_debug,
        verbose=True,
    )
    print(
        f"D. Matched {len(matches)} undirected edge(s) "
        f"(adaptive max endpoint dist={match_debug.max_endpoint_distance:.1f}px, "
        f"{len(match_debug.rejected)} rejected)."
    )
    for m in matches:
        print(
            f"   {m.node_a} -- {m.node_b}  "
            f"(arrow {m.arrow_index}, d={m.dist_a:.1f}/{m.dist_b:.1f})"
        )

    debug_path = PROJECT_ROOT / "data" / "debug" / "phase3_relationship_debug.png"
    labels = {n["id"]: str(n.get("label", "")) for n in shape_nodes}
    render_relationship_debug(
        image,
        shape_refs,
        loc.raw_lines,
        loc.lines,
        matches,
        debug=match_debug,
        path=debug_path,
        labels=labels,
    )
    print(f"   debug overlay: {debug_path}")

    # E. NetworkX graph
    graph_nodes = [
        {
            "id": n["id"],
            "shape_type": n["shape_type"],
            "label": n["label"],
            "bbox": n["bbox"],
            "confidence": n["confidence"],
        }
        for n in shape_nodes
    ]
    graph = build_graph(graph_nodes, matches)
    print(f"E. Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # F. Log + JSON
    print("\nF. Summary")
    print("  Nodes:")
    for nid, attrs in sorted(graph.nodes(data=True)):
        print(
            f"    id={nid}  type={attrs['shape_type']:8s}  "
            f"label={attrs['label']!r}  conf={attrs['confidence']:.3f}  bbox={attrs['bbox']}"
        )
    print("  Edges:")
    if graph.number_of_edges() == 0:
        print("    (none)")
    for u, v in sorted(tuple(sorted(e)) for e in graph.edges()):
        print(f"    {u} -- {v}")

    payload = save_graph_json(
        graph,
        args.output,
        extra={"source": source, "image_size": [w, h]},
    )
    print(f"\nWrote {args.output}")
    print("DOT:\n" + payload["dot"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
