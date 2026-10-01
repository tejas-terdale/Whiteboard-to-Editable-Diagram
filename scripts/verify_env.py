#!/usr/bin/env python3
"""Environment verification for the Whiteboard-to-Diagram project.

Checks Python version, PyTorch/CUDA, OpenCV, and the Graphviz `dot` binary.
Exits with a non-zero status if a hard requirement is missing.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Callable

# Windows + conda/pip often loads two OpenMP runtimes (OpenCV and PyTorch).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SEPARATOR = "-" * 60


def _ok(label: str, detail: str) -> None:
    print(f"[OK]   {label}: {detail}")


def _warn(label: str, detail: str) -> None:
    print(f"[WARN] {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"[FAIL] {label}: {detail}")


def check_python(min_major: int = 3, min_minor: int = 9) -> bool:
    """Require Python >= 3.9."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) >= (min_major, min_minor):
        _ok("Python", f"{version_str} ({sys.executable})")
        return True
    _fail(
        "Python",
        f"{version_str} found; need >= {min_major}.{min_minor}",
    )
    return False


def check_pytorch() -> bool:
    """Confirm torch is importable and report GPU/CUDA status."""
    try:
        import torch
    except ImportError as exc:
        _fail("PyTorch", f"not installed ({exc})")
        return False

    detail = f"torch {torch.__version__}"
    cuda_built = torch.cuda.is_available()
    if cuda_built:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        detail += (
            f" | CUDA available | GPU: {device_name} "
            f"(compute {capability[0]}.{capability[1]})"
        )
        if torch.version.cuda:
            detail += f" | CUDA toolkit {torch.version.cuda}"
        _ok("PyTorch", detail)
    else:
        _warn(
            "PyTorch",
            f"{detail} | CUDA not available — CPU training only",
        )
    return True


def check_opencv() -> bool:
    """Confirm OpenCV imports and report build metadata."""
    try:
        import cv2
    except ImportError as exc:
        _fail("OpenCV", f"not installed ({exc})")
        return False

    build = cv2.getBuildInformation()
    # First line of the build dump usually includes version / config.
    first_line = next(
        (line.strip() for line in build.splitlines() if line.strip()),
        "build info unavailable",
    )
    _ok("OpenCV", f"cv2 {cv2.__version__} | {first_line}")
    return True


def check_graphviz_binary() -> bool:
    """Check that the Graphviz `dot` executable is on PATH."""
    dot_path = shutil.which("dot")
    if dot_path:
        _ok("Graphviz", f"`dot` found at {dot_path}")
        return True
    _fail(
        "Graphviz",
        "`dot` is not on PATH. Install Graphviz system package "
        "(e.g. winget install Graphviz.Graphviz) and restart the shell.",
    )
    return False


def check_optional_imports() -> None:
    """Soft-check remaining requirements.txt packages (warnings only)."""
    packages: list[tuple[str, Callable[[], object]]] = [
        ("numpy", lambda: __import__("numpy")),
        ("PIL", lambda: __import__("PIL")),
        ("matplotlib", lambda: __import__("matplotlib")),
        ("networkx", lambda: __import__("networkx")),
        ("graphviz (python)", lambda: __import__("graphviz")),
        ("streamlit", lambda: __import__("streamlit")),
        ("easyocr", lambda: __import__("easyocr")),
        ("torchvision", lambda: __import__("torchvision")),
    ]
    print(SEPARATOR)
    print("Optional package import check (from requirements.txt)")
    print(SEPARATOR)
    for name, loader in packages:
        try:
            module = loader()
            version = getattr(module, "__version__", "ok")
            _ok(name, str(version))
        except ImportError as exc:
            _warn(name, f"not importable ({exc})")


def main() -> int:
    print(SEPARATOR)
    print("Whiteboard-to-Diagram - environment verification")
    print(SEPARATOR)

    results = [
        check_python(),
        check_pytorch(),
        check_opencv(),
        check_graphviz_binary(),
    ]
    check_optional_imports()

    print(SEPARATOR)
    if all(results):
        print("Result: all hard checks passed.")
        return 0

    print("Result: one or more hard checks failed. See [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
