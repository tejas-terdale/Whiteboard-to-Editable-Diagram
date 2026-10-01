"""Geometry-only tests for Hough connectors and undirected endpoint matching."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.localization import localize
from src.relationships import (
    ArrowRef,
    ShapeRef,
    arrow_ref_from_segment,
    match_arrows_to_shapes,
)


def _board_with_two_boxes() -> np.ndarray:
    board = np.full((400, 700, 3), 245, dtype=np.uint8)
    ink = (20, 20, 20)
    cv2.rectangle(board, (40, 140), (200, 260), ink, 3)
    cv2.rectangle(board, (480, 140), (640, 260), ink, 3)
    cv2.line(board, (210, 200), (460, 200), ink, 3)
    return board


def test_hough_finds_connector_between_boxes() -> None:
    image = _board_with_two_boxes()
    loc = localize(image)
    assert len(loc.shapes) >= 2
    assert len(loc.lines) >= 1
    lengths = [s.length for s in loc.lines]
    assert max(lengths) > 150


def test_match_connects_the_two_boxes() -> None:
    image = _board_with_two_boxes()
    loc = localize(image)
    shapes = [
        ShapeRef(node_id=i, bbox=c.bbox, contour=c.contour)
        for i, c in enumerate(loc.shapes)
    ]
    arrows = [arrow_ref_from_segment(s) for s in loc.lines]
    matches = match_arrows_to_shapes(arrows, shapes, image_size=image.shape[:2])
    pairs = {(m.node_a, m.node_b) for m in matches}
    assert len(pairs) >= 1
    assert len(pairs) == len(matches)


def test_duplicate_segments_collapse_to_one_edge() -> None:
    shapes = [
        ShapeRef(node_id=0, bbox=(0, 0, 40, 40)),
        ShapeRef(node_id=1, bbox=(200, 0, 40, 40)),
    ]
    arrows = [
        ArrowRef(contour=np.array([[[45, 20]], [[195, 20]]], dtype=np.float32)),
        ArrowRef(contour=np.array([[[50, 22]], [[190, 18]]], dtype=np.float32)),
    ]
    matches = match_arrows_to_shapes(arrows, shapes, max_endpoint_distance=30)
    assert len(matches) == 1
    assert (matches[0].node_a, matches[0].node_b) == (0, 1)


def test_self_loop_and_internal_line_dropped() -> None:
    shapes = [ShapeRef(node_id=0, bbox=(0, 0, 100, 80))]
    arrows = [
        ArrowRef(contour=np.array([[[5, 10]], [[90, 10]]], dtype=np.float32)),
    ]
    matches = match_arrows_to_shapes(arrows, shapes, max_endpoint_distance=20)
    assert matches == []


if __name__ == "__main__":
    test_hough_finds_connector_between_boxes()
    test_match_connects_the_two_boxes()
    test_duplicate_segments_collapse_to_one_edge()
    test_self_loop_and_internal_line_dropped()
    print("test_line_detection: all passed")
