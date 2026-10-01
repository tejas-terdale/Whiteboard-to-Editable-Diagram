"""Shared end-to-end pipeline: localize → classify → OCR → match → NetworkX."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.graph_builder import build_graph
from src.localization import localize, render_line_debug
from src.model import classify_bgr_crops, load_checkpoint
from src.ocr import SHAPE_CLASSES, ShapeOCR
from src.relationships import (
    ArrowRef,
    MatchDebug,
    ShapeRef,
    collect_arrow_refs,
    match_arrows_to_shapes,
    render_relationship_debug,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "shape_classifier.pth"


@dataclass
class PipelineResult:
    graph: Any
    n_localized_shapes: int
    n_localized_arrows: int
    n_nodes: int
    n_arrows_classified: int
    n_edges: int
    log_lines: list[str] = field(default_factory=list)
    n_hough_lines: int = 0
    debug_image_path: str | None = None


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
        nested = any(
            i != j and _bbox_contained(node["bbox"], other["bbox"])
            for j, other in enumerate(nodes)
        )
        if not nested:
            survivors.append(node)
    for new_id, node in enumerate(survivors):
        node["id"] = new_id
    return survivors


def load_classifier(weights: Path | None = None, device: torch.device | None = None):
    path = weights or DEFAULT_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing classifier weights at {path}. "
            "Train first: python scripts/train_classifier.py"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(path, device=device)
    return model, device


def run_pipeline(
    image: np.ndarray,
    model=None,
    device: torch.device | None = None,
    ocr: ShapeOCR | None = None,
    weights: Path | None = None,
    ocr_min_confidence: float = 0.25,
    ocr_gpu: bool = False,
    debug_dir: Path | None = None,
    skip_ocr: bool = False,
) -> PipelineResult:
    """Run Stages 1–6 on a BGR whiteboard photo and return a NetworkX graph."""
    log: list[str] = []
    if image is None or image.size == 0:
        raise ValueError("Empty image.")

    if model is None:
        model, device = load_classifier(weights, device)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc = localize(image)
    log.append(
        f"Localized {len(loc.shapes)} compact and {len(loc.arrows)} elongated contour(s); "
        f"{len(loc.raw_lines)} raw Hough line(s) merged to {len(loc.lines)} connector(s)."
    )

    preds = classify_bgr_crops([c.crop for c in loc.all_candidates], model, device)
    shape_nodes: list[dict] = []
    arrow_refs: list[ArrowRef] = []
    node_id = 0
    for cand, (label, conf) in zip(loc.all_candidates, preds):
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
    n_contour_arrows = len(arrow_refs)
    arrow_refs = collect_arrow_refs(arrow_refs, loc.lines)

    before = len(shape_nodes)
    shape_nodes = drop_nested_shape_nodes(shape_nodes)
    if len(shape_nodes) != before:
        log.append(f"Dropped {before - len(shape_nodes)} nested crop(s) (interior text).")

    log.append(
        f"Classifier kept {len(shape_nodes)} shape node(s) and {n_contour_arrows} contour arrow(s); "
        f"matching uses {len(arrow_refs)} shaft candidate(s) including Hough lines."
    )

    if not skip_ocr:
        engine = ocr or ShapeOCR(min_confidence=ocr_min_confidence, gpu=ocr_gpu)
        for node in shape_nodes:
            result = engine.read_shape(node["crop"], node["id"])
            node["label"] = result.text
            log.append(
                f"OCR node {node['id']} ({node['shape_type']}): {result.text!r}"
                f"{' [fallback]' if result.used_fallback else ''}"
            )

    h, w = image.shape[:2]
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
        verbose=False,
    )
    log.append(
        f"Matched {len(matches)} undirected edge(s) "
        f"(max endpoint dist={match_debug.max_endpoint_distance:.1f}px, "
        f"rejected={len(match_debug.rejected)})."
    )

    debug_path = None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "relationship_debug.png"
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
        log.append(f"Wrote relationship debug overlay to {debug_path}")
        line_path = debug_dir / "line_debug.png"
        render_line_debug(image, loc, path=line_path)
        log.append(f"Wrote line-detection debug panel to {line_path}")

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
    log.append(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    return PipelineResult(
        graph=graph,
        n_localized_shapes=len(loc.shapes),
        n_localized_arrows=len(loc.arrows),
        n_nodes=graph.number_of_nodes(),
        n_arrows_classified=n_contour_arrows,
        n_edges=graph.number_of_edges(),
        log_lines=log,
        n_hough_lines=len(loc.lines),
        debug_image_path=str(debug_path) if debug_path else None,
    )


def build_labeled_demo_whiteboard() -> np.ndarray:
    """Worked-example board used when no photo is uploaded."""
    board = np.full((640, 1100, 3), 242, dtype=np.uint8)
    yy = np.linspace(0, 10, board.shape[0], dtype=np.float32)
    board = np.clip(board.astype(np.int16) + yy[:, None, None].astype(np.int16), 0, 255)
    board = board.astype(np.uint8)
    ink = (25, 25, 25)
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
        unit = vec / (np.linalg.norm(vec) + 1e-6)
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
