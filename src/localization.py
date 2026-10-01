"""Classical CV localization: preprocess, find contours, split shape vs arrow.

Stage 1 (preprocess) and Stage 2 (contour filtering) of the pipeline.
No learned detector is used — candidates are purely geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
# Wide process boxes (e.g. "Provide Service") are still nodes.
_SHAPE_ASPECT_FILLED_MAX = 5.2
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
class HoughParams:
    """``cv2.HoughLinesP`` settings used for the last detection pass."""

    rho: float = 1.0
    theta: float = float(np.pi / 180.0)
    threshold: int = 12
    min_line_length: int = 12
    max_line_gap: int = 18


@dataclass
class LineSegment:
    """Straight connector from ``cv2.HoughLinesP`` (optionally merged)."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    source: str = "hough"

    @property
    def length(self) -> float:
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x1, y1 = self.p1
        x2, y2 = self.p2
        x = int(min(x1, x2))
        y = int(min(y1, y2))
        w = max(1, int(abs(x2 - x1)) + 1)
        h = max(1, int(abs(y2 - y1)) + 1)
        return (x, y, w, h)

    def as_contour(self) -> np.ndarray:
        return np.array([[self.p1], [self.p2]], dtype=np.float32)


@dataclass
class LocalizationResult:
    """All candidates plus the binary image used for contour finding."""

    binary: np.ndarray
    shapes: list[Candidate]
    arrows: list[Candidate]
    lines: list[LineSegment] = field(default_factory=list)
    raw_lines: list[LineSegment] = field(default_factory=list)
    hough_params: HoughParams = field(default_factory=HoughParams)
    line_ink: np.ndarray | None = None

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


def _looks_filled_node(m: dict[str, float]) -> bool:
    """Closed boxes/circles/diamonds, including wide process rectangles."""
    aspect = max(m["aspect_rot"], m["aspect_bbox"])
    return (
        m["solidity"] >= _SHAPE_SOLIDITY_MIN
        and m["extent"] >= 0.32
        and aspect <= _SHAPE_ASPECT_FILLED_MAX
    )


def _kind_from_metrics(m: dict[str, float]) -> CandidateKind | None:
    """Map geometry to shape (node) vs arrow (edge), or reject as noise."""
    aspect = max(m["aspect_rot"], m["aspect_bbox"])
    solidity = m["solidity"]
    extent = m["extent"]

    elongated = aspect >= _ARROW_ASPECT_MIN
    compact = aspect <= _SHAPE_ASPECT_MAX and solidity >= _SHAPE_SOLIDITY_MIN
    filled_enough = extent >= _SHAPE_EXTENT_MIN

    # Filled rectangles must not be treated as arrows or Hough traces their borders.
    if _looks_filled_node(m):
        return "shape"
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

    raw_lines, lines, hough_params, line_ink = detect_connector_lines(binary, shapes)
    return LocalizationResult(
        binary=binary,
        shapes=shapes,
        arrows=arrows,
        lines=lines,
        raw_lines=raw_lines,
        hough_params=hough_params,
        line_ink=line_ink,
    )


def localize(image: np.ndarray) -> LocalizationResult:
    """Full Stage 1+2 entry point: preprocess then extract candidates."""
    _gray, _blurred, binary = preprocess(image)
    return extract_candidates(image, binary=binary)


def _estimate_stroke_width(binary: np.ndarray) -> float:
    """Approximate marker width in pixels from the ink distance transform."""
    ink = binary if binary.max() > 1 else (binary * 255).astype(np.uint8)
    if cv2.countNonZero(ink) < 20:
        return 3.0
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    vals = dist[ink > 0]
    if vals.size == 0:
        return 3.0
    half = float(np.percentile(vals, 80))
    return float(np.clip(max(2.0, 2.0 * half), 2.0, 8.0))


def _is_bloated_shape(m: dict[str, float], img_area: float) -> bool:
    """True when a 'shape' contour swallowed neighboring nodes/arrows."""
    if m["bbox_area"] > 0.07 * img_area:
        return True
    aspect = max(m["aspect_rot"], m["aspect_bbox"])
    return aspect >= 3.8 and m["extent"] < 0.38


def _is_compact_node(m: dict[str, float], img_area: float) -> bool:
    if _is_bloated_shape(m, img_area):
        return False
    aspect = max(m["aspect_rot"], m["aspect_bbox"])
    return (
        m["solidity"] >= 0.50
        and m["extent"] >= 0.22
        and aspect <= _SHAPE_ASPECT_FILLED_MAX
        and m["area"] >= max(120.0, 0.0004 * img_area)
    )


def _looks_like_text(m: dict[str, float], img_area: float) -> bool:
    """Small compact blobs (letters, 'Yes'/'No', watermarks) — not shafts."""
    aspect = max(m["aspect_rot"], m["aspect_bbox"])
    if m["area"] > max(900.0, 0.004 * img_area):
        return False
    if aspect >= 4.0:
        return False
    return m["solidity"] >= 0.40 and m["extent"] >= 0.20


def _shape_fill_mask(binary_shape: tuple[int, int], shapes: list[Candidate]) -> np.ndarray:
    """Filled compact nodes only — bloated merged contours are skipped."""
    mask = np.zeros(binary_shape, dtype=np.uint8)
    img_area = float(binary_shape[0] * binary_shape[1])
    for shape in shapes:
        if _is_bloated_shape(shape.metrics, img_area):
            continue
        if shape.contour is None or len(shape.contour) < 3:
            x, y, w, h = shape.bbox
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(mask, [shape.contour], -1, 255, thickness=cv2.FILLED)
    return mask


def _hole_and_node_mask(binary: np.ndarray) -> np.ndarray:
    """Fill hollow-shape interiors (CCOMP holes) and isolated compact nodes.

    Attached arrows can glue several nodes into one EXTERNAL contour. Holes
    inside each box/diamond/circle still mark the true node interior, so
    Hough can hide borders without erasing neighboring connectors.
    """
    h, w = binary.shape[:2]
    img_area = float(h * w)
    mask = np.zeros((h, w), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return mask
    hierarchy = hierarchy[0]
    min_hole = max(180.0, 0.0008 * img_area)
    for i, contour in enumerate(contours):
        metrics = contour_metrics(contour)
        if metrics is None:
            continue
        parent = int(hierarchy[i][3])
        if parent != -1:
            if min_hole <= metrics["area"] < 0.35 * img_area:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            continue
        if _is_compact_node(metrics, img_area):
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def _non_connector_mask(binary: np.ndarray, shapes: list[Candidate]) -> np.ndarray:
    """Hide node interiors, compact boxes, and text so Hough sees only shafts."""
    mask = cv2.bitwise_or(_hole_and_node_mask(binary), _shape_fill_mask(binary.shape[:2], shapes))
    # Erode shape interior mask slightly so outer stroke boundaries stay in ink
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    
    img_area = float(binary.shape[0] * binary.shape[1])
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra = np.zeros_like(mask)
    for contour in contours:
        metrics = contour_metrics(contour)
        if metrics is None:
            continue
        if _looks_like_text(metrics, img_area):
            cv2.drawContours(extra, [contour], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_or(mask, extra)


def _is_page_or_grid_line(seg: LineSegment, img_w: int, img_h: int) -> bool:
    """Filter out image frame / page border lines and long canvas grid lines."""
    x1, y1 = seg.p1
    x2, y2 = seg.p2
    margin = 15

    near_top = (y1 <= margin and y2 <= margin)
    near_bottom = (y1 >= img_h - margin and y2 >= img_h - margin)
    near_left = (x1 <= margin and x2 <= margin)
    near_right = (x1 >= img_w - margin and x2 >= img_w - margin)

    if (near_top or near_bottom or near_left or near_right) and seg.length > 0.22 * min(img_w, img_h):
        return True

    # Long straight line near borders
    if (x1 < margin or x2 < margin or x1 > img_w - margin or x2 > img_w - margin) and abs(y2 - y1) > 0.40 * img_h:
        return True
    if (y1 < margin or y2 < margin or y1 > img_h - margin or y2 > img_h - margin) and abs(x2 - x1) > 0.40 * img_w:
        return True

    # Spans across > 45% of width or height horizontally/vertically near image edge
    if abs(x2 - x1) > 0.45 * img_w and abs(y2 - y1) < 12 and (y1 < 25 or y1 > img_h - 25):
        return True
    if abs(y2 - y1) > 0.45 * img_h and abs(x2 - x1) < 12 and (x1 < 25 or x1 > img_w - 25):
        return True

    return False


def _is_on_shape_border(seg: LineSegment, shapes: list[Candidate]) -> bool:
    """True if the line segment's midpoint lies directly on a shape contour/border."""
    if not shapes:
        return False
    mid = ((seg.p1[0] + seg.p2[0]) * 0.5, (seg.p1[1] + seg.p2[1]) * 0.5)
    for s in shapes:
        if s.contour is not None and len(s.contour) >= 3:
            dist = abs(cv2.pointPolygonTest(s.contour, mid, measureDist=True))
            if dist <= 5.0:
                return True
        else:
            x, y, w, h = s.bbox
            if (x - 5 <= mid[0] <= x + w + 5) and (y - 5 <= mid[1] <= y + h + 5):
                on_horiz = (abs(mid[1] - y) <= 5) or (abs(mid[1] - (y + h)) <= 5)
                on_vert = (abs(mid[0] - x) <= 5) or (abs(mid[0] - (x + w)) <= 5)
                if on_horiz or on_vert:
                    return True
    return False


def _hough_params(binary: np.ndarray) -> HoughParams:
    """Scale votes/length/gap to image size for typical flowchart shafts."""
    min_side = float(min(binary.shape[:2]))
    threshold = max(8, int(0.010 * min_side))
    min_len = max(10, int(0.014 * min_side))
    max_gap = max(12, int(0.028 * min_side))
    return HoughParams(
        rho=1.0,
        theta=float(np.pi / 180.0),
        threshold=int(threshold),
        min_line_length=int(min_len),
        max_line_gap=int(max_gap),
    )


def _segments_from_hough(ink: np.ndarray, params: HoughParams) -> list[LineSegment]:
    detected = cv2.HoughLinesP(
        ink,
        rho=params.rho,
        theta=params.theta,
        threshold=params.threshold,
        minLineLength=params.min_line_length,
        maxLineGap=params.max_line_gap,
    )
    if detected is None:
        return []
    lines: list[LineSegment] = []
    min_keep = params.min_line_length * 0.75
    for row in np.asarray(detected).reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in row)
        seg = LineSegment(p1=(x1, y1), p2=(x2, y2), source="hough")
        if seg.length >= min_keep:
            lines.append(seg)
    return lines


def _angle(seg: LineSegment) -> float:
    return float(np.arctan2(seg.p2[1] - seg.p1[1], seg.p2[0] - seg.p1[0]))


def _angle_delta(a: float, b: float) -> float:
    d = abs(a - b) % np.pi
    return float(min(d, np.pi - d))


def _point_to_infinite_line(point: tuple[float, float], seg: LineSegment) -> float:
    x0, y0 = point
    x1, y1 = seg.p1
    x2, y2 = seg.p2
    dx, dy = x2 - x1, y2 - y1
    denom = float(np.hypot(dx, dy))
    if denom < 1e-6:
        return float(np.hypot(x0 - x1, y0 - y1))
    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom


def _endpoint_gap(a: LineSegment, b: LineSegment) -> float:
    pts_a = (a.p1, a.p2)
    pts_b = (b.p1, b.p2)
    return min(float(np.hypot(p[0] - q[0], p[1] - q[1])) for p in pts_a for q in pts_b)


def _projections_overlap_or_near(
    a: LineSegment, b: LineSegment, gap: float
) -> bool:
    axis_ang = _angle(a)
    axis = np.array([np.cos(axis_ang), np.sin(axis_ang)], dtype=np.float64)
    origin = np.array(a.p1, dtype=np.float64)

    def _proj(pt: tuple[float, float]) -> float:
        return float((np.array(pt, dtype=np.float64) - origin) @ axis)

    a0, a1 = sorted((_proj(a.p1), _proj(a.p2)))
    b0, b1 = sorted((_proj(b.p1), _proj(b.p2)))
    if a1 < b0:
        return (b0 - a1) <= gap
    if b1 < a0:
        return (a0 - b1) <= gap
    return True


def _merge_two(a: LineSegment, b: LineSegment) -> LineSegment:
    pts = np.array([a.p1, a.p2, b.p1, b.p2], dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    p_lo = pts[int(np.argmin(proj))]
    p_hi = pts[int(np.argmax(proj))]
    return LineSegment(
        p1=(float(p_lo[0]), float(p_lo[1])),
        p2=(float(p_hi[0]), float(p_hi[1])),
        source="merged",
    )


def merge_collinear_segments(
    segments: list[LineSegment],
    angle_deg: float = 15.0,
    dist_px: float | None = None,
    gap_px: float | None = None,
) -> list[LineSegment]:
    """Join nearby, nearly collinear Hough fragments into longer connectors."""
    if not segments:
        return []
    lengths = [s.length for s in segments]
    typical = float(np.median(lengths)) if lengths else 40.0
    if dist_px is None:
        dist_px = max(8.0, 0.18 * typical)
    if gap_px is None:
        gap_px = max(20.0, 0.40 * typical)
    ang_th = np.deg2rad(angle_deg)

    remaining = list(segments)
    changed = True
    while changed:
        changed = False
        used = [False] * len(remaining)
        nxt: list[LineSegment] = []
        for i, si in enumerate(remaining):
            if used[i]:
                continue
            cur = si
            used[i] = True
            for j in range(i + 1, len(remaining)):
                if used[j]:
                    continue
                sj = remaining[j]
                if _angle_delta(_angle(cur), _angle(sj)) > ang_th:
                    continue
                off = max(
                    _point_to_infinite_line(sj.p1, cur),
                    _point_to_infinite_line(sj.p2, cur),
                    _point_to_infinite_line(cur.p1, sj),
                    _point_to_infinite_line(cur.p2, sj),
                )
                if off > dist_px:
                    continue
                if not _projections_overlap_or_near(cur, sj, gap_px):
                    continue
                cur = _merge_two(cur, sj)
                used[j] = True
                changed = True
            nxt.append(cur)
        remaining = nxt
    return remaining


def _endpoint_close(
    p: tuple[float, float], q: tuple[float, float], join_px: float
) -> bool:
    return float(np.hypot(p[0] - q[0], p[1] - q[1])) <= join_px


def chain_polyline_segments(
    segments: list[LineSegment], join_px: float | None = None
) -> list[LineSegment]:
    """Join collinear fragments that share an endpoint (not L / T bends).

    Perpendicular flowchart elbows are left as separate segments so the next
    phase can match each shaft end to a node independently.
    """
    if len(segments) < 2:
        return list(segments)
    if join_px is None:
        lengths = [s.length for s in segments]
        join_px = max(12.0, 0.18 * float(np.median(lengths)))

    n = len(segments)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[rj] = ri

    ends = [(s.p1, s.p2) for s in segments]
    ang_th = np.deg2rad(12.0)
    for i in range(n):
        for j in range(i + 1, n):
            if _angle_delta(_angle(segments[i]), _angle(segments[j])) > ang_th:
                continue
            linked = any(
                _endpoint_close(p, q, join_px)
                for p in ends[i]
                for q in ends[j]
            )
            if linked:
                _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    chained: list[LineSegment] = []
    consumed: set[int] = set()
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        cur = segments[idxs[0]]
        for k in idxs[1:]:
            cur = _merge_two(cur, segments[k])
        chained.append(LineSegment(p1=cur.p1, p2=cur.p2, source="chained"))
        consumed.update(idxs)

    out = [s for i, s in enumerate(segments) if i not in consumed]
    out.extend(chained)
    return out


def _segment_hits_mask(
    seg: LineSegment, mask: np.ndarray, frac: float = 0.55, n: int = 20
) -> bool:
    """True if most samples along the segment sit on the node/text mask."""
    h, w = mask.shape[:2]
    xs = np.linspace(seg.p1[0], seg.p2[0], n)
    ys = np.linspace(seg.p1[1], seg.p2[1], n)
    hits = 0
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
            hits += 1
    return hits / float(n) >= frac


def _drop_near_duplicates(
    segments: list[LineSegment], dist_px: float = 6.0, angle_deg: float = 8.0
) -> list[LineSegment]:
    """Keep the longer of two nearly coincident Hough hits."""
    if not segments:
        return []
    ang_th = np.deg2rad(angle_deg)
    kept: list[LineSegment] = []
    for seg in sorted(segments, key=lambda s: s.length, reverse=True):
        dup = False
        for other in kept:
            if _angle_delta(_angle(seg), _angle(other)) > ang_th:
                continue
            off = max(
                _point_to_infinite_line(seg.p1, other),
                _point_to_infinite_line(seg.p2, other),
            )
            if off > dist_px:
                continue
            if _projections_overlap_or_near(seg, other, dist_px):
                dup = True
                break
        if not dup:
            kept.append(seg)
    return kept


def detect_connector_lines(
    binary: np.ndarray,
    shapes: list[Candidate],
) -> tuple[list[LineSegment], list[LineSegment], HoughParams, np.ndarray]:
    """Hough line detector that does not alter contour-based shape results.

    Returns ``(raw_segments, merged_segments, params, ink_used)``. Node
    interiors (including CCOMP holes) are masked so rectangle/circle
    outlines are not treated as connectors.
    """
    hide = _non_connector_mask(binary, shapes)
    ink = cv2.bitwise_and(binary, cv2.bitwise_not(hide))
    # Bridge tiny gaps in shafts without thickening into nearby nodes.
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_k, iterations=1)

    params = _hough_params(ink)
    # Sensitive pass for short ticks and fragmented connectors
    short = HoughParams(
        rho=params.rho,
        theta=params.theta,
        threshold=max(6, params.threshold - 2),
        min_line_length=max(8, int(params.min_line_length * 0.7)),
        max_line_gap=params.max_line_gap + 4,
    )

    # Convert ink to 1-px Canny edges for precise Hough axis detection
    edges = cv2.Canny(ink_closed, 50, 150)
    raw = _segments_from_hough(edges, params) + _segments_from_hough(edges, short)

    img_h, img_w = binary.shape[:2]
    # Filter out page borders and shape outlines
    filtered_raw = []
    for s in raw:
        if _is_page_or_grid_line(s, img_w, img_h):
            continue
        if _is_on_shape_border(s, shapes):
            continue
        filtered_raw.append(s)

    filtered_raw = _drop_near_duplicates(filtered_raw, dist_px=5.0, angle_deg=6.0)
    merged = merge_collinear_segments(filtered_raw, angle_deg=12.0, dist_px=10.0, gap_px=30.0)
    merged = chain_polyline_segments(merged, join_px=15.0)

    min_side = min(img_h, img_w)
    min_keep = float(max(8.0, 0.010 * min_side))
    merged = [s for s in merged if s.length >= min_keep]
    merged = [s for s in merged if not _is_page_or_grid_line(s, img_w, img_h)]
    merged = [s for s in merged if not _is_on_shape_border(s, shapes)]
    merged = _drop_near_duplicates(merged, dist_px=8.0, angle_deg=8.0)
    return filtered_raw, merged, params, ink


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

    hp = result.hough_params
    return {
        "n_shapes": len(result.shapes),
        "n_arrows": len(result.arrows),
        "n_lines": len(result.lines),
        "n_raw_lines": len(result.raw_lines),
        "hough_params": {
            "rho": hp.rho,
            "theta": hp.theta,
            "threshold": hp.threshold,
            "minLineLength": hp.min_line_length,
            "maxLineGap": hp.max_line_gap,
        },
        "shapes": _pack(result.shapes),
        "arrows": _pack(result.arrows),
    }


def render_line_debug(
    image: np.ndarray,
    result: LocalizationResult,
    path: str | Path | None = None,
) -> np.ndarray:
    """Panel: original, detected shapes, raw Hough, merged connectors.

    Writes a *new* file; ``image`` is never modified in place.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()
    h, w = bgr.shape[:2]

    shapes_panel = bgr.copy()
    for shape in result.shapes:
        cv2.rectangle(
            shapes_panel,
            (shape.bbox[0], shape.bbox[1]),
            (shape.bbox[0] + shape.bbox[2], shape.bbox[1] + shape.bbox[3]),
            (40, 180, 40),
            2,
        )
        if shape.contour is not None and len(shape.contour) >= 3:
            cv2.drawContours(shapes_panel, [shape.contour], -1, (0, 160, 0), 1)

    raw_panel = bgr.copy()
    for seg in result.raw_lines:
        p1 = (int(round(seg.p1[0])), int(round(seg.p1[1])))
        p2 = (int(round(seg.p2[0])), int(round(seg.p2[1])))
        cv2.line(raw_panel, p1, p2, (220, 200, 40), 2, cv2.LINE_AA)

    merged_panel = bgr.copy()
    for seg in result.lines:
        p1 = (int(round(seg.p1[0])), int(round(seg.p1[1])))
        p2 = (int(round(seg.p2[0])), int(round(seg.p2[1])))
        cv2.line(merged_panel, p1, p2, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(merged_panel, p1, 4, (0, 220, 255), -1)
        cv2.circle(merged_panel, p2, 4, (0, 220, 255), -1)

    def _caption(panel: np.ndarray, text: str) -> np.ndarray:
        out = panel.copy()
        cv2.rectangle(out, (0, 0), (w, 28), (255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(
            out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA
        )
        return out

    hp = result.hough_params
    top = np.hstack(
        (
            _caption(bgr, "1. original"),
            _caption(shapes_panel, f"2. shapes ({len(result.shapes)})"),
        )
    )
    bottom = np.hstack(
        (
            _caption(
                raw_panel,
                f"3. raw Hough ({len(result.raw_lines)})  "
                f"th={hp.threshold} minLen={hp.min_line_length} gap={hp.max_line_gap}",
            ),
            _caption(merged_panel, f"4. merged connectors ({len(result.lines)})"),
        )
    )
    canvas = np.vstack((top, bottom))

    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
    return canvas
