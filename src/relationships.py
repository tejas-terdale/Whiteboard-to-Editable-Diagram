"""Undirected geometric matching: arrow endpoints → nearest shape nodes.

No learned model and no arrow-head / direction detection (Stage 5 spec).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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


def match_arrows_to_shapes(
    arrows: list[ArrowRef],
    shapes: list[ShapeRef],
    max_endpoint_distance: float | None = None,
) -> list[EdgeMatch]:
    """Pair each arrow's two endpoints with the nearest shape nodes.

    Undirected: ``(a, b)`` is stored with ``node_a < node_b``. Self-loops and
    duplicate pairs are dropped. If ``max_endpoint_distance`` is set, an
    endpoint farther than that is treated as unmatched (edge discarded).
    """
    seen: set[tuple[int, int]] = set()
    matches: list[EdgeMatch] = []

    for idx, arrow in enumerate(arrows):
        end_a, end_b = arrow_endpoints(arrow.contour)
        hit_a = nearest_shape(end_a, shapes)
        hit_b = nearest_shape(end_b, shapes)
        if hit_a is None or hit_b is None:
            continue

        shape_a, dist_a = hit_a
        shape_b, dist_b = hit_b
        if max_endpoint_distance is not None:
            if dist_a > max_endpoint_distance or dist_b > max_endpoint_distance:
                continue
        if shape_a.node_id == shape_b.node_id:
            continue

        lo, hi = sorted((shape_a.node_id, shape_b.node_id))
        pair = (lo, hi)
        if pair in seen:
            continue
        seen.add(pair)
        matches.append(
            EdgeMatch(
                node_a=lo,
                node_b=hi,
                arrow_index=idx,
                endpoint_a=end_a,
                endpoint_b=end_b,
                dist_a=dist_a,
                dist_b=dist_b,
            )
        )
    return matches
