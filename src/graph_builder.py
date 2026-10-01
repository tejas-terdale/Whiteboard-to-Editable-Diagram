"""Build an undirected NetworkX graph from classified shapes, OCR labels, and edges."""

from __future__ import annotations

import json
from typing import Any, Iterable

import networkx as nx

from src.relationships import EdgeMatch

GRAPHVIZ_SHAPES = {
    "box": "box",
    "circle": "ellipse",
    "diamond": "diamond",
}


def build_graph(
    nodes: Iterable[dict[str, Any]],
    edges: Iterable[EdgeMatch] | Iterable[tuple[int, int]],
) -> nx.Graph:
    """Create ``nx.Graph`` with node attrs: shape_type, label, bbox, confidence."""
    graph = nx.Graph()
    for node in nodes:
        node_id = int(node["id"])
        bbox = node.get("bbox", (0, 0, 0, 0))
        if hasattr(bbox, "tolist"):
            bbox = tuple(int(v) for v in bbox.tolist())
        else:
            bbox = tuple(int(v) for v in bbox)
        graph.add_node(
            node_id,
            shape_type=str(node.get("shape_type", "box")),
            label=str(node.get("label", f"Node_{node_id}")),
            bbox=bbox,
            confidence=float(node.get("confidence", 0.0)),
        )

    for edge in edges:
        if isinstance(edge, EdgeMatch):
            a, b = edge.node_a, edge.node_b
        else:
            a, b = int(edge[0]), int(edge[1])
        if a == b:
            continue
        if not graph.has_node(a) or not graph.has_node(b):
            continue
        if graph.has_edge(a, b):
            continue
        graph.add_edge(int(a), int(b))
    return graph


def graph_to_dict(graph: nx.Graph) -> dict[str, Any]:
    """JSON-friendly snapshot of nodes and undirected edges."""
    nodes = []
    for node_id, attrs in sorted(graph.nodes(data=True)):
        nodes.append(
            {
                "id": int(node_id),
                "shape_type": attrs.get("shape_type"),
                "label": attrs.get("label"),
                "bbox": list(attrs.get("bbox", ())),
                "confidence": float(attrs.get("confidence", 0.0)),
            }
        )
    edges = [
        {"source": int(u), "target": int(v)}
        for u, v in sorted(tuple(sorted(e)) for e in graph.edges())
    ]
    return {"nodes": nodes, "edges": edges}


def _escape_dot_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def graph_to_dot(graph: nx.Graph, graph_name: str = "whiteboard") -> str:
    """Manual Graphviz DOT (undirected). Avoids a pydot extra dependency."""
    lines = [f"graph {graph_name} {{", "  rankdir=LR;", '  node [fontname="Helvetica"];']
    for node_id, attrs in sorted(graph.nodes(data=True)):
        label = _escape_dot_label(str(attrs.get("label", f"Node_{node_id}")))
        gv_shape = GRAPHVIZ_SHAPES.get(str(attrs.get("shape_type", "box")), "box")
        lines.append(
            f'  n{node_id} [label="{label}", shape={gv_shape}];'
        )
    for u, v in sorted(tuple(sorted(e)) for e in graph.edges()):
        lines.append(f"  n{u} -- n{v};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def save_graph_json(
    graph: nx.Graph,
    path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write graph dict (+ optional metadata and DOT) to ``path``."""
    payload = graph_to_dict(graph)
    payload["dot"] = graph_to_dot(graph)
    if extra:
        payload.update(extra)
    path = path if hasattr(path, "write_text") else __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
