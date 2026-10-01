"""Convert a NetworkX whiteboard graph into Graphviz DOT for Streamlit."""

from __future__ import annotations

import networkx as nx

from src.graph_builder import GRAPHVIZ_SHAPES, graph_to_dot

FILL = {
    "box": "#DBEAFE",
    "circle": "#DCFCE7",
    "diamond": "#FEF3C7",
}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def graph_to_styled_dot(graph: nx.Graph, graph_name: str = "whiteboard") -> str:
    """Undirected DOT with Graphviz shapes mapped from ``shape_type``."""
    lines = [
        f"graph {graph_name} {{",
        "  rankdir=LR;",
        '  bgcolor="white";',
        '  node [fontname="Helvetica", style=filled, color="#334155"];',
        '  edge [color="#64748B", penwidth=1.6];',
    ]
    for node_id, attrs in sorted(graph.nodes(data=True)):
        label = _escape(str(attrs.get("label", f"Node_{node_id}")))
        shape_type = str(attrs.get("shape_type", "box"))
        gv_shape = GRAPHVIZ_SHAPES.get(shape_type, "box")
        fill = FILL.get(shape_type, "#F1F5F9")
        lines.append(
            f'  n{node_id} [label="{label}", shape={gv_shape}, fillcolor="{fill}"];'
        )
    for u, v in sorted(tuple(sorted(e)) for e in graph.edges()):
        lines.append(f"  n{u} -- n{v};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_dot(graph: nx.Graph | None, styled: bool = True) -> str:
    """Return DOT for the active graph, or an empty placeholder."""
    if graph is None or graph.number_of_nodes() == 0:
        return 'graph whiteboard {\n  empty [label="No diagram yet", shape=box, style=dashed];\n}\n'
    if styled:
        return graph_to_styled_dot(graph)
    return graph_to_dot(graph)


def st_draw_graph(graph: nx.Graph | None) -> None:
    """Render the diagram inside Streamlit (``st.graphviz_chart``)."""
    import streamlit as st

    st.graphviz_chart(to_dot(graph), use_container_width=True)
