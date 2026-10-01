"""OCR on shape crops only (box / circle / diamond), never arrows or the full board.

EasyOCR is the implemented engine (spec fallback when PaddleOCR is not used).
Low-confidence or empty reads become ``Node_{id}``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SHAPE_CLASSES = frozenset({"box", "circle", "diamond"})
_MIN_SIDE = 160
_DEFAULT_MIN_CONF = 0.25
_FALLBACK = "Node_{id}"


@dataclass
class OCRResult:
    text: str
    confidence: float
    used_fallback: bool


def preprocess_shape_crop(crop: np.ndarray) -> np.ndarray:
    """Upscale small crops and boost contrast for marker-style writing."""
    if crop.ndim == 2:
        bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        bgr = crop.copy()

    h, w = bgr.shape[:2]
    scale = max(1.0, _MIN_SIDE / float(min(h, w)))
    if scale > 1.01:
        bgr = cv2.resize(
            bgr,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    luminance, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    luminance = clahe.apply(luminance)
    enhanced = cv2.cvtColor(cv2.merge([luminance, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    return enhanced


def _clean_text(raw: str) -> str:
    collapsed = re.sub(r"\s+", " ", raw).strip()
    return collapsed


class ShapeOCR:
    """Lazy EasyOCR wrapper so importing ``src.ocr`` does not download models."""

    def __init__(
        self,
        languages: list[str] | None = None,
        min_confidence: float = _DEFAULT_MIN_CONF,
        gpu: bool | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.min_confidence = min_confidence
        self.gpu = gpu
        self._reader = None
        self._unavailable_reason: str | None = None

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        if self._unavailable_reason:
            return None
        try:
            import easyocr
            import torch
        except ImportError as exc:
            self._unavailable_reason = str(exc)
            print(f"[ocr] EasyOCR not available ({exc}); using fallback labels.")
            return None

        # CPU is the default: EasyOCR on CUDA often deadlocks with OpenCV/PyTorch
        # OpenMP on Windows. A handful of shape crops is fast enough on CPU.
        use_gpu = False if self.gpu is None else self.gpu
        print(f"[ocr] Loading EasyOCR (gpu={use_gpu})...", flush=True)
        self._reader = easyocr.Reader(self.languages, gpu=use_gpu, verbose=False)
        print("[ocr] EasyOCR ready.", flush=True)
        return self._reader

    def read_shape(self, crop: np.ndarray, node_id: int) -> OCRResult:
        """Return cleaned text or ``Node_{id}`` if OCR fails / is too unsure."""
        fallback = _FALLBACK.format(id=node_id)
        reader = self._get_reader()
        if reader is None:
            return OCRResult(text=fallback, confidence=0.0, used_fallback=True)

        prepared = preprocess_shape_crop(crop)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        try:
            detections = reader.readtext(rgb, detail=1, paragraph=False)
        except Exception as exc:  # EasyOCR/runtime glitches should not kill the pipeline
            print(f"[ocr] read failed for node {node_id}: {exc}")
            return OCRResult(text=fallback, confidence=0.0, used_fallback=True)

        snippets: list[str] = []
        confidences: list[float] = []
        for item in detections:
            if not item or len(item) < 3:
                continue
            text, conf = str(item[1]), float(item[2])
            if conf < self.min_confidence:
                continue
            cleaned = _clean_text(text)
            if cleaned:
                snippets.append(cleaned)
                confidences.append(conf)

        joined = _clean_text(" ".join(snippets))
        if not joined:
            return OCRResult(text=fallback, confidence=0.0, used_fallback=True)
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return OCRResult(text=joined, confidence=mean_conf, used_fallback=False)


def label_shape_crops(
    crops: list[np.ndarray],
    node_ids: list[int],
    engine: ShapeOCR | None = None,
) -> list[OCRResult]:
    """OCR each shape crop; ``node_ids`` are used only for fallback names."""
    ocr = engine or ShapeOCR()
    return [ocr.read_shape(crop, nid) for crop, nid in zip(crops, node_ids)]
