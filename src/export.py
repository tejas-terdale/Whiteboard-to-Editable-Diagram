"""Export the edited NetworkX graph as PNG or PDF via Graphviz."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import networkx as nx

from src.renderer import to_dot

_DOT_TIMEOUT_SEC = 20


class ExportError(RuntimeError):
    """Raised when the Graphviz binary cannot render."""


def _dot_executable() -> str:
    exe = shutil.which("dot")
    if not exe:
        raise ExportError(
            "Graphviz `dot` is not on PATH. Install Graphviz and restart the app."
        )
    return exe


def _via_dot_binary(dot: str, fmt: str) -> bytes:
    """Render through the system `dot` binary (stdin → stdout), with a timeout.

    Avoids ``graphviz.Source.pipe()``, which can hang indefinitely on Windows.
    """
    exe = _dot_executable()
    try:
        proc = subprocess.run(
            [exe, f"-T{fmt}"],
            input=dot.encode("utf-8"),
            capture_output=True,
            timeout=_DOT_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExportError(
            f"Graphviz `dot` timed out after {_DOT_TIMEOUT_SEC}s while rendering {fmt}."
        ) from exc
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ExportError(err or f"dot failed to render {fmt}.")
    return proc.stdout


def export_bytes(graph: nx.Graph | None, fmt: str = "png") -> bytes:
    """Return PNG or PDF bytes for ``graph`` using Graphviz."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError(f"Unsupported export format: {fmt}")
    if graph is None or graph.number_of_nodes() == 0:
        raise ExportError("Nothing to export — process a photo or add nodes first.")
    return _via_dot_binary(to_dot(graph, styled=True), fmt)


def export_to_file(graph: nx.Graph | None, path: Path, fmt: str | None = None) -> Path:
    """Write a PNG/PDF next to ``path`` (extension inferred if ``fmt`` omitted)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "png").lower()
    data = export_bytes(graph, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
