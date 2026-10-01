"""Undirected geometric matching: arrow endpoints → nearest shape nodes.

No learned model and no arrow-head / direction detection (Stage 5 spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from src.localization import LineSegment


@dataclass
class ShapeRef:
    node_id: int
    bbox: tuple[int, int, int, int]
    contour: np.ndarray | None = None


@dataclass
class ArrowRef:
    contour: np.ndarray
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class EdgeMatch:
    node_a: int
    node_b: int
    arrow_index: int
    endpoint_a: tuple[float, float]
    endpoint_b: tuple[float, float]
    dist_a: float
    dist_b: float


@dataclass
class RejectedMatch:
    arrow_index: int
    endpoint_a: tuple[float, float]
    endpoint_b: tuple[float, float]
    node_a: int | None
    node_b: int | None
    dist_a: float
    dist_b: float
    reason: str


@dataclass
class MatchDebug:
    """Raw pairings plus rejects, filled when matching runs with debug=True."""

    raw_pairings: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[RejectedMatch] = field(default_factory=list)
    max_endpoint_distance: float = 0.0


def arrow_ref_from_segment(segment: LineSegment) -> ArrowRef:
    """Turn a Hough/merged segment into the ArrowRef the matcher already uses."""
    return ArrowRef(contour=segment.as_contour(), bbox=segment.bbox)


def arrow_endpoints(contour: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two extreme points of the contour along its principal axis."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        c = pts[0] if len(pts) else np.zeros(2)
        xy = (float(c[0]), float(c[1]))
        return xy, xy

    centered = pts - pts.mean(axis=0)
    # SVD principal direction; fall back to minAreaRect if the stroke is tiny.
    if np.linalg.norm(centered) < 1e-6:
        return _endpoints_from_min_area_rect(contour)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def _endpoints_from_min_area_rect(
    contour: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    theta = np.deg2rad(angle)
    dx, dy = np.cos(theta), np.sin(theta)
    half = rw / 2.0
    a = (float(cx - dx * half), float(cy - dy * half))
    b = (float(cx + dx * half), float(cy + dy * half))
    return a, b


def point_to_bbox_distance(
    point: tuple[float, float],
    bbox: tuple[int, int, int, int],
) -> float:
    """0 if the point is inside the box; else distance to the nearest edge/corner."""
    px, py = point
    x, y, w, h = bbox
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return float(np.hypot(dx, dy))


def point_to_shape_distance(point: tuple[float, float], shape: ShapeRef) -> float:
    """Prefer contour distance; bbox distance if no contour is stored."""
    if shape.contour is not None and len(shape.contour) >= 3:
        dist = cv2.pointPolygonTest(
            shape.contour,
            (float(point[0]), float(point[1])),
            measureDist=True,
        )
        if dist >= 0:
            return 0.0
        return float(abs(dist))
    return point_to_bbox_distance(point, shape.bbox)


def nearest_shape(
    point: tuple[float, float],
    shapes: list[ShapeRef],
) -> tuple[ShapeRef, float] | None:
    if not shapes:
        return None
    best = shapes[0]
    best_d = point_to_shape_distance(point, best)
    for shape in shapes[1:]:
        d = point_to_shape_distance(point, shape)
        if d < best_d:
            best, best_d = shape, d
    return best, best_d


def adaptive_max_endpoint_distance(
    shapes: list[ShapeRef],
    image_size: tuple[int, int] | None = None,
) -> float:
    """Gap allowed between a connector tip and a node boundary.

    Uses shape size and image diagonal so short flowchart arrows still match
    when they stop just outside a box, without a single tight pixel cutoff.
    """
    if image_size is not None:
        h, w = image_size
        diag = float(np.hypot(w, h))
    elif shapes:
        max_x = max(s.bbox[0] + s.bbox[2] for s in shapes)
        max_y = max(s.bbox[1] + s.bbox[3] for s in shapes)
        diag = float(np.hypot(max_x, max_y))
    else:
        diag = 800.0

    if shapes:
        min_sides = [float(min(s.bbox[2], s.bbox[3])) for s in shapes]
        median_min = float(np.median(min_sides))
    else:
        median_min = 60.0

    return max(40.0, 0.07 * diag, 0.45 * median_min)


def _sample_segment(
    p1: tuple[float, float], p2: tuple[float, float], n: int = 24
) -> list[tuple[float, float]]:
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _internal_to_single_shape(
    p1: tuple[float, float],
    p2: tuple[float, float],
    shapes: list[ShapeRef],
    frac: float = 0.65,
) -> ShapeRef | None:
    """True when most of the stroke sits inside one node (outline leftover)."""
    samples = _sample_segment(p1, p2)
    if not samples or not shapes:
        return None
    best_shape = None
    best_hits = 0
    for shape in shapes:
        hits = sum(1 for pt in samples if point_to_shape_distance(pt, shape) <= 1.5)
        if hits > best_hits:
            best_hits = hits
            best_shape = shape
    if best_shape is not None and best_hits / len(samples) >= frac:
        return best_shape
    return None


def _crosses_unrelated_shape(
    p1: tuple[float, float],
    p2: tuple[float, float],
    node_a: int,
    node_b: int,
    shapes: list[ShapeRef],
) -> bool:
    mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    hit = nearest_shape(mid, shapes)
    if hit is None:
        return False
    shape, dist = hit
    if dist > 1.0:
        return False
    return shape.node_id not in (node_a, node_b)


def match_arrows_to_shapes(
    arrows: list[ArrowRef],
    shapes: list[ShapeRef],
    max_endpoint_distance: float | None = None,
    image_size: tuple[int, int] | None = None,
    debug: MatchDebug | None = None,
    verbose: bool = False,
) -> list[EdgeMatch]:
    """Pair each arrow's two endpoints with the nearest shape nodes.

    Undirected: ``(a, b)`` is stored with ``node_a < node_b``. Self-loops and
    duplicate pairs are dropped. Handles single straight line segments as well
    as 2-segment L-bend connectors that meet at an elbow joint.
    """
    if max_endpoint_distance is None:
        max_endpoint_distance = adaptive_max_endpoint_distance(shapes, image_size)

    if debug is not None:
        debug.max_endpoint_distance = float(max_endpoint_distance)
        debug.raw_pairings.clear()
        debug.rejected.clear()

    # Collect single-segment and 2-segment L-bend path candidates
    candidates: list[dict[str, Any]] = []
    
    # 1. Single segments
    for idx, arrow in enumerate(arrows):
        end_a, end_b = arrow_endpoints(arrow.contour)
        hit_a = nearest_shape(end_a, shapes)
        hit_b = nearest_shape(end_b, shapes)
        
        node_a = hit_a[0].node_id if hit_a else None
        node_b = hit_b[0].node_id if hit_b else None
        dist_a = hit_a[1] if hit_a else float("inf")
        dist_b = hit_b[1] if hit_b else float("inf")

        candidates.append({
            "arrow_index": idx,
            "end_a": end_a,
            "end_b": end_b,
            "hit_a": hit_a,
            "hit_b": hit_b,
            "node_a": node_a,
            "node_b": node_b,
            "dist_a": dist_a,
            "dist_b": dist_b,
            "cost": dist_a + dist_b,
            "is_lbend": False,
        })

    # 2. Two-segment L-bends
    n_arrows = len(arrows)
    for i in range(n_arrows):
        end_a1, end_a2 = arrow_endpoints(arrows[i].contour)
        for j in range(i + 1, n_arrows):
            end_b1, end_b2 = arrow_endpoints(arrows[j].contour)
            combos = [
                (end_a1, end_a2, end_b1, end_b2),
                (end_a1, end_a2, end_b2, end_b1),
                (end_a2, end_a1, end_b1, end_b2),
                (end_a2, end_a1, end_b2, end_b1),
            ]
            for p_start, p_join1, p_join2, p_end in combos:
                join_d = float(np.hypot(p_join1[0] - p_join2[0], p_join1[1] - p_join2[1]))
                if join_d <= 22.0:
                    hit_a = nearest_shape(p_start, shapes)
                    hit_b = nearest_shape(p_end, shapes)
                    if hit_a and hit_b:
                        sa, da = hit_a
                        sb, db = hit_b
                        candidates.append({
                            "arrow_index": i,
                            "end_a": p_start,
                            "end_b": p_end,
                            "hit_a": hit_a,
                            "hit_b": hit_b,
                            "node_a": sa.node_id,
                            "node_b": sb.node_id,
                            "dist_a": da,
                            "dist_b": db,
                            "cost": da + db + join_d * 0.5,
                            "is_lbend": True,
                        })

    # Sort candidate pairings by cost (best / lowest distance first)
    candidates.sort(key=lambda item: item["cost"])

    seen: dict[tuple[int, int], EdgeMatch] = {}
    matches: list[EdgeMatch] = []

    def _reject(
        idx: int,
        end_a: tuple[float, float],
        end_b: tuple[float, float],
        node_a: int | None,
        node_b: int | None,
        dist_a: float,
        dist_b: float,
        reason: str,
    ) -> None:
        if debug is not None:
            debug.rejected.append(
                RejectedMatch(
                    arrow_index=idx,
                    endpoint_a=end_a,
                    endpoint_b=end_b,
                    node_a=node_a,
                    node_b=node_b,
                    dist_a=dist_a,
                    dist_b=dist_b,
                    reason=reason,
                )
            )

    for cand in candidates:
        idx = cand["arrow_index"]
        end_a = cand["end_a"]
        end_b = cand["end_b"]
        hit_a = cand["hit_a"]
        hit_b = cand["hit_b"]
        dist_a = cand["dist_a"]
        dist_b = cand["dist_b"]
        node_a = cand["node_a"]
        node_b = cand["node_b"]

        pairing = {
            "arrow_index": idx,
            "endpoint_a": end_a,
            "endpoint_b": end_b,
            "node_a": node_a,
            "node_b": node_b,
            "dist_a": None if hit_a is None else round(dist_a, 2),
            "dist_b": None if hit_b is None else round(dist_b, 2),
            "is_lbend": cand["is_lbend"],
        }
        if debug is not None:
            debug.raw_pairings.append(pairing)

        if hit_a is None or hit_b is None:
            _reject(idx, end_a, end_b, node_a, node_b, dist_a, dist_b, "missing_shape")
            continue

        shape_a, dist_a = hit_a
        shape_b, dist_b = hit_b
        if dist_a > max_endpoint_distance or dist_b > max_endpoint_distance:
            _reject(
                idx, end_a, end_b, shape_a.node_id, shape_b.node_id, dist_a, dist_b,
                "endpoint_too_far",
            )
            continue
        if shape_a.node_id == shape_b.node_id:
            _reject(
                idx, end_a, end_b, shape_a.node_id, shape_b.node_id, dist_a, dist_b,
                "self_loop",
            )
            continue

        internal = _internal_to_single_shape(end_a, end_b, shapes)
        if internal is not None:
            _reject(
                idx, end_a, end_b, shape_a.node_id, shape_b.node_id, dist_a, dist_b,
                f"internal_to_node_{internal.node_id}",
            )
            continue

        if _crosses_unrelated_shape(end_a, end_b, shape_a.node_id, shape_b.node_id, shapes):
            _reject(
                idx, end_a, end_b, shape_a.node_id, shape_b.node_id, dist_a, dist_b,
                "crosses_unrelated_shape",
            )
            continue

        lo, hi = sorted((shape_a.node_id, shape_b.node_id))
        pair = (lo, hi)
        existing = seen.get(pair)
        if existing is not None:
            _reject(
                idx, end_a, end_b, lo, hi, dist_a, dist_b, "duplicate_pair"
            )
            continue

        match = EdgeMatch(
            node_a=lo,
            node_b=hi,
            arrow_index=idx,
            endpoint_a=end_a,
            endpoint_b=end_b,
            dist_a=dist_a,
            dist_b=dist_b,
        )
        seen[pair] = match
        matches.append(match)
    return matches


def collect_arrow_refs(
    contour_arrows: Iterable[ArrowRef],
    line_segments: Iterable[LineSegment],
) -> list[ArrowRef]:
    """CNN/contour arrows plus dedicated Hough connectors, one list for matching."""
    refs = list(contour_arrows)
    for seg in line_segments:
        refs.append(arrow_ref_from_segment(seg))
    return refs


def render_relationship_debug(
    image: np.ndarray,
    shapes: list[ShapeRef],
    raw_lines: list[LineSegment],
    lines: list[LineSegment],
    matches: list[EdgeMatch],
    debug: MatchDebug | None = None,
    path: str | Path | None = None,
    labels: dict[int, str] | None = None,
) -> np.ndarray:
    """Overlay shapes, Hough segments, endpoints, and accepted/rejected edges."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    # Shapes: green boxes + ids
    for shape in shapes:
        x, y, w, h = shape.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 180, 40), 2)
        name = labels.get(shape.node_id, "") if labels else ""
        caption = f"{shape.node_id}" if not name else f"{shape.node_id}:{name[:18]}"
        cv2.putText(
            canvas,
            caption,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 120, 20),
            1,
            cv2.LINE_AA,
        )
        if shape.contour is not None and len(shape.contour) >= 3:
            cv2.drawContours(canvas, [shape.contour.astype(np.int32)], -1, (40, 160, 40), 1)

    # Raw Hough: cyan
    for seg in raw_lines:
        p1 = (int(round(seg.p1[0])), int(round(seg.p1[1])))
        p2 = (int(round(seg.p2[0])), int(round(seg.p2[1])))
        cv2.line(canvas, p1, p2, (220, 200, 40), 1, cv2.LINE_AA)

    # Merged connectors: orange
    for seg in lines:
        p1 = (int(round(seg.p1[0])), int(round(seg.p1[1])))
        p2 = (int(round(seg.p2[0])), int(round(seg.p2[1])))
        cv2.line(canvas, p1, p2, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 220, 255), -1)
        cv2.circle(canvas, p2, 4, (0, 220, 255), -1)

    if debug is not None:
        for rej in debug.rejected:
            if rej.reason in {"duplicate_pair", "internal_to_node"}:
                color = (80, 80, 200)
            else:
                color = (0, 0, 220)
            p1 = (int(round(rej.endpoint_a[0])), int(round(rej.endpoint_a[1])))
            p2 = (int(round(rej.endpoint_b[0])), int(round(rej.endpoint_b[1])))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    # Accepted relationships: thick green + node pair label
    for match in matches:
        p1 = (int(round(match.endpoint_a[0])), int(round(match.endpoint_a[1])))
        p2 = (int(round(match.endpoint_b[0])), int(round(match.endpoint_b[1])))
        cv2.line(canvas, p1, p2, (0, 200, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, (0, 255, 0), 2)
        cv2.circle(canvas, p2, 6, (0, 255, 0), 2)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(
            canvas,
            f"{match.node_a}-{match.node_b}",
            (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 100, 0),
            1,
            cv2.LINE_AA,
        )

    legend_y = 22
    for text, color in (
        ("green box = shape", (40, 180, 40)),
        ("cyan = raw Hough", (220, 200, 40)),
        ("orange = merged line", (0, 140, 255)),
        ("green edge = accepted", (0, 200, 0)),
        ("red = rejected", (0, 0, 220)),
    ):
        cv2.putText(
            canvas, text, (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        legend_y += 18

    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
    return canvas
