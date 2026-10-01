"""Classical CV localization: preprocess, find contours, split shape vs arrow.

Stage 1 (preprocess) and Stage 2 (contour filtering) of the pipeline.
No learned detector is used — candidates are purely geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

CandidateKind = Literal["shape", "arrow"]

# Drop specks; keep as a fraction of image area so it scales with resolution.
_MIN_AREA_FRAC = 0.00025
_MIN_AREA_ABS = 80
# Ignore a contour that covers most of the frame (page border, full whiteboard).
_MAX_AREA_FRAC = 0.55

# Compact closed nodes vs thin elongated edges.
_SHAPE_ASPECT_MAX = 2.6
_SHAPE_SOLIDITY_MIN = 0.55
_SHAPE_EXTENT_MIN = 0.25
_ARROW_ASPECT_MIN = 2.2
_ARROW_SOLIDITY_MAX = 0.82

_CROP_PAD = 12
_APPROX_EPS_FRAC = 0.02


@dataclass
class Candidate:
    """One localized region to send to the CNN (and later OCR / matching)."""

    kind: CandidateKind
    bbox: tuple[int, int, int, int]  # x, y, w, h in original image coords
    crop: np.ndarray  # BGR crop, padded
    contour: np.ndarray
    approx: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class LocalizationResult:
    """All candidates plus the binary image used for contour finding."""

    binary: np.ndarray
    shapes: list[Candidate]
    arrows: list[Candidate]

    @property
    def all_candidates(self) -> list[Candidate]:
        return self.shapes + self.arrows


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grayscale → Gaussian blur → adaptive threshold (ink as white).

    Returns ``(gray, blurred, binary)``. Binary is ``uint8`` {0, 255} with
    marker strokes as 255 so ``findContours`` traces ink, not the board.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape[:2]
    block = 31 if min(h, w) >= 400 else 21
    if block % 2 == 0:
        block += 1

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, blurred, binary


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 1e-6 else 0.0


def contour_metrics(contour: np.ndarray) -> dict[str, float] | None:
    """Geometric features used to drop noise and split shape vs arrow."""
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h)
    peri = float(cv2.arcLength(contour, closed=True))
    approx = cv2.approxPolyDP(contour, _APPROX_EPS_FRAC * peri, closed=True)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    rw, rh = float(rw), float(rh)

    aspect_bbox = _safe_div(max(w, h), min(w, h))
    aspect_rot = _safe_div(max(rw, rh), min(rw, rh))

    return {
        "area": area,
        "bbox_x": float(x),
        "bbox_y": float(y),
        "bbox_w": float(w),
        "bbox_h": float(h),
        "bbox_area": bbox_area,
        "extent": _safe_div(area, bbox_area),
        "solidity": _safe_div(area, hull_area),
        "aspect_bbox": aspect_bbox,
        "aspect_rot": aspect_rot,
        "perimeter": peri,
        "approx_vertices": float(len(approx)),
        "min_rect_w": rw,
        "min_rect_h": rh,
    }


def _kind_from_metrics(m: dict[str, float]) -> CandidateKind | None:
    """Map geometry to shape (node) vs arrow (edge), or reject as noise."""
    aspect = max(m["aspect_rot"], m["aspect_bbox"])
    solidity = m["solidity"]
    extent = m["extent"]

    elongated = aspect >= _ARROW_ASPECT_MIN
    compact = aspect <= _SHAPE_ASPECT_MAX and solidity >= _SHAPE_SOLIDITY_MIN
    filled_enough = extent >= _SHAPE_EXTENT_MIN

    if elongated and solidity <= _ARROW_SOLIDITY_MAX:
        return "arrow"
    if elongated and not compact:
        return "arrow"
    if compact and filled_enough:
        return "shape"
    if elongated:
        return "arrow"
    return None


def _crop_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int = _CROP_PAD,
) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=image.dtype) if image.ndim == 3 else np.zeros((1, 1), dtype=image.dtype)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return crop


def extract_candidates(
    image: np.ndarray,
    binary: np.ndarray | None = None,
) -> LocalizationResult:
    """Find contours on ``binary`` (or preprocess ``image``) and split them.

    ``image`` should be the original BGR photo so crops look like the CNN's
    synthetic training data (gray-white board + dark stroke).
    """
    if binary is None:
        _, _, binary = preprocess(image)

    img_h, img_w = binary.shape[:2]
    img_area = float(img_h * img_w)
    min_area = max(_MIN_AREA_ABS, _MIN_AREA_FRAC * img_area)
    max_area = _MAX_AREA_FRAC * img_area

    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    shapes: list[Candidate] = []
    arrows: list[Candidate] = []

    for contour in contours:
        metrics = contour_metrics(contour)
        if metrics is None:
            continue
        if metrics["area"] < min_area or metrics["area"] > max_area:
            continue

        kind = _kind_from_metrics(metrics)
        if kind is None:
            continue

        peri = metrics["perimeter"]
        approx = cv2.approxPolyDP(contour, _APPROX_EPS_FRAC * peri, closed=True)
        x, y, w, h = (
            int(metrics["bbox_x"]),
            int(metrics["bbox_y"]),
            int(metrics["bbox_w"]),
            int(metrics["bbox_h"]),
        )
        bbox = (x, y, w, h)
        candidate = Candidate(
            kind=kind,
            bbox=bbox,
            crop=_crop_bbox(image, bbox),
            contour=contour,
            approx=approx,
            metrics=metrics,
        )
        if kind == "shape":
            shapes.append(candidate)
        else:
            arrows.append(candidate)

    return LocalizationResult(binary=binary, shapes=shapes, arrows=arrows)


def localize(image: np.ndarray) -> LocalizationResult:
    """Full Stage 1+2 entry point: preprocess then extract candidates."""
    _gray, _blurred, binary = preprocess(image)
    return extract_candidates(image, binary=binary)


def result_as_jsonable(result: LocalizationResult) -> dict[str, Any]:
    """Metadata-only dump (no image arrays) for logging."""
    def _pack(cands: list[Candidate]) -> list[dict[str, Any]]:
        packed = []
        for c in cands:
            packed.append(
                {
                    "kind": c.kind,
                    "bbox": c.bbox,
                    "approx_vertices": int(c.metrics.get("approx_vertices", 0)),
                    "metrics": {k: round(v, 4) for k, v in c.metrics.items()},
                }
            )
        return packed

    return {
        "n_shapes": len(result.shapes),
        "n_arrows": len(result.arrows),
        "shapes": _pack(result.shapes),
        "arrows": _pack(result.arrows),
    }
