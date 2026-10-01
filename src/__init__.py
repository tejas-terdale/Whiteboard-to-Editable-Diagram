"""Whiteboard photo to editable diagram — shared source package."""

from src.export import export_bytes
from src.graph_builder import build_graph, graph_to_dict, graph_to_dot
from src.localization import Candidate, LocalizationResult, localize, preprocess
from src.model import CLASS_NAMES, build_model, load_checkpoint
from src.ocr import ShapeOCR
from src.pipeline import run_pipeline
from src.relationships import match_arrows_to_shapes
from src.renderer import to_dot

__all__ = [
    "CLASS_NAMES",
    "Candidate",
    "LocalizationResult",
    "ShapeOCR",
    "build_graph",
    "build_model",
    "export_bytes",
    "graph_to_dict",
    "graph_to_dot",
    "load_checkpoint",
    "localize",
    "match_arrows_to_shapes",
    "preprocess",
    "run_pipeline",
    "to_dot",
]
