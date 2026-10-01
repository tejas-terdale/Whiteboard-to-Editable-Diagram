#!/usr/bin/env python3
"""Streamlit UI: upload a whiteboard photo, reconstruct, edit, and export."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import cv2
import numpy as np
import streamlit as st
import torch

from src.export import ExportError, export_bytes
from src.pipeline import (
    DEFAULT_WEIGHTS,
    build_labeled_demo_whiteboard,
    load_classifier,
    run_pipeline,
)
from src.ocr import ShapeOCR
from src.renderer import st_draw_graph, to_dot

st.set_page_config(
    page_title="Whiteboard to Editable Diagram",
    page_icon="✏️",
    layout="wide",
)

SHAPE_CHOICES = ("box", "circle", "diamond")


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_upload(uploaded) -> np.ndarray | None:
    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return image


def _init_state() -> None:
    defaults = {
        "graph": None,
        "image_bgr": None,
        "pipeline_log": [],
        "status": "Waiting for an image.",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner="Loading MobileNetV2 classifier...")
def _cached_classifier():
    return load_classifier()


@st.cache_resource(show_spinner="Loading EasyOCR (first run can take a minute)...")
def _cached_ocr():
    return ShapeOCR(min_confidence=0.25, gpu=False)


def _node_label(graph, node_id: int) -> str:
    attrs = graph.nodes[node_id]
    return f"{node_id}: {attrs.get('label', '')} ({attrs.get('shape_type', 'box')})"


def _next_node_id(graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    return int(max(graph.nodes)) + 1


def _apply_crop(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0, x1 = max(0, min(left, right)), min(w, max(left, right))
    y0, y1 = max(0, min(top, bottom)), min(h, max(top, bottom))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return image
    return image[y0:y1, x0:x1]


def _sidebar_status() -> None:
    st.sidebar.header("Pipeline status")
    weights_ok = DEFAULT_WEIGHTS.is_file()
    st.sidebar.write(f"{'✅' if weights_ok else '❌'} Classifier weights")
    if not weights_ok:
        st.sidebar.caption(f"Expected at `{DEFAULT_WEIGHTS}`")
    cuda = torch.cuda.is_available()
    st.sidebar.write(f"{'✅' if cuda else 'ℹ️'} CUDA {'available' if cuda else 'not available (CPU OK)'}")
    graph = st.session_state.graph
    if graph is None:
        st.sidebar.write("⬜ Diagram not built yet")
    else:
        st.sidebar.write(
            f"✅ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to use")
    st.sidebar.markdown(
        "1. Upload a PNG/JPG whiteboard photo (or load the demo).\n"
        "2. Optionally crop, then click **Process Diagram**.\n"
        "3. Rename, add, or delete nodes/edges with the form widgets.\n"
        "4. Download PNG or PDF from this sidebar."
    )


def _graph_fingerprint(graph) -> tuple:
    nodes = tuple(
        (int(n), str(attrs.get("label")), str(attrs.get("shape_type")))
        for n, attrs in sorted(graph.nodes(data=True))
    )
    edges = tuple(sorted(tuple(sorted(e)) for e in graph.edges()))
    return nodes, edges


def _export_buttons() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export")
    graph = st.session_state.graph
    disabled = graph is None or graph.number_of_nodes() == 0
    if disabled:
        st.sidebar.caption("Process a photo first to enable export.")
        return

    sig = _graph_fingerprint(graph)
    if st.session_state.get("export_sig") != sig:
        try:
            st.session_state.export_png = export_bytes(graph, "png")
            st.session_state.export_pdf = export_bytes(graph, "pdf")
            st.session_state.export_sig = sig
            st.session_state.export_error = None
        except ExportError as exc:
            st.session_state.export_error = str(exc)
            st.session_state.export_sig = None

    if st.session_state.get("export_error"):
        st.sidebar.error(st.session_state.export_error)
        return

    st.sidebar.download_button(
        "Download PNG",
        data=st.session_state.export_png,
        file_name="whiteboard_diagram.png",
        mime="image/png",
        use_container_width=True,
    )
    st.sidebar.download_button(
        "Download PDF",
        data=st.session_state.export_pdf,
        file_name="whiteboard_diagram.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    with st.sidebar.expander("DOT source"):
        st.code(to_dot(graph), language="dot")


def _editor_panel() -> None:
    graph = st.session_state.graph
    st.subheader("Edit diagram")
    if graph is None:
        st.info("Process a photo to enable editing.")
        return

    node_ids = sorted(graph.nodes)
    node_options = [_node_label(graph, n) for n in node_ids]
    id_by_label = dict(zip(node_options, node_ids))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Rename node**")
        if node_ids:
            pick = st.selectbox("Node", node_options, key="rename_pick")
            nid = id_by_label[pick]
            new_label = st.text_input(
                "New label",
                value=str(graph.nodes[nid].get("label", "")),
                key=f"rename_text_{nid}",
            )
            if st.button("Update label", key="btn_rename"):
                graph.nodes[nid]["label"] = new_label.strip() or f"Node_{nid}"
                st.session_state.graph = graph
                st.rerun()
        else:
            st.caption("No nodes yet.")

        st.markdown("**Delete node**")
        if node_ids:
            del_pick = st.selectbox("Node to delete", node_options, key="delete_pick")
            if st.button("Delete node and its edges", key="btn_del_node"):
                graph.remove_node(id_by_label[del_pick])
                st.session_state.graph = graph
                st.rerun()

    with col_b:
        st.markdown("**Add node**")
        add_shape = st.selectbox("Shape type", SHAPE_CHOICES, key="add_shape")
        add_label = st.text_input("Label", value="New node", key="add_label")
        connect_choices = ["(none)"] + node_options
        connect_pick = st.selectbox("Connect to", connect_choices, key="add_connect")
        if st.button("Add node", key="btn_add_node"):
            nid = _next_node_id(graph)
            graph.add_node(
                nid,
                shape_type=add_shape,
                label=add_label.strip() or f"Node_{nid}",
                bbox=(0, 0, 0, 0),
                confidence=1.0,
            )
            if connect_pick != "(none)":
                graph.add_edge(nid, id_by_label[connect_pick])
            st.session_state.graph = graph
            st.rerun()

        st.markdown("**Add edge**")
        if len(node_ids) >= 2:
            e1 = st.selectbox("From", node_options, key="edge_a")
            e2 = st.selectbox("To", node_options, key="edge_b")
            if st.button("Add undirected edge", key="btn_add_edge"):
                a, b = id_by_label[e1], id_by_label[e2]
                if a == b:
                    st.warning("Pick two different nodes.")
                else:
                    graph.add_edge(a, b)
                    st.session_state.graph = graph
                    st.rerun()
        else:
            st.caption("Need at least two nodes to add an edge.")

        st.markdown("**Delete edge**")
        edges = [tuple(sorted(e)) for e in graph.edges()]
        if edges:
            edge_labels = [
                f"{u} ({graph.nodes[u].get('label')}) — {v} ({graph.nodes[v].get('label')})"
                for u, v in edges
            ]
            edge_pick = st.selectbox("Edge", edge_labels, key="del_edge_pick")
            if st.button("Delete edge", key="btn_del_edge"):
                u, v = edges[edge_labels.index(edge_pick)]
                graph.remove_edge(u, v)
                st.session_state.graph = graph
                st.rerun()
        else:
            st.caption("No edges to delete.")


def main() -> None:
    _init_state()

    st.title("Whiteboard Photo to Editable Diagram")
    st.caption(
        "Upload a reasonably straight-on whiteboard photo. "
        "Shapes and arrows are localized with OpenCV, classified with MobileNetV2, "
        "labeled with EasyOCR, and assembled into an undirected Graphviz diagram."
    )

    _sidebar_status()
    _export_buttons()

    top_l, top_r = st.columns([2, 1])
    with top_l:
        uploaded = st.file_uploader(
            "Upload whiteboard photo",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
    with top_r:
        st.write("")
        st.write("")
        if st.button("Load demo whiteboard", use_container_width=True):
            st.session_state.image_bgr = build_labeled_demo_whiteboard()
            st.session_state.graph = None
            st.session_state.pipeline_log = []
            st.session_state.status = "Demo board loaded. Click Process Diagram."
            st.rerun()

    if uploaded is not None:
        decoded = _decode_upload(uploaded)
        if decoded is None:
            st.error("Could not decode that file. Try another PNG or JPG.")
        else:
            st.session_state.image_bgr = decoded

    image = st.session_state.image_bgr
    work = image
    if image is not None:
        h, w = image.shape[:2]
        st.subheader("Photo preview")
        preview_l, preview_r = st.columns(2)
        with preview_l:
            st.markdown("**Original**")
            st.image(_bgr_to_rgb(image), use_container_width=True)
        with preview_r:
            st.markdown("**Manual crop (optional)**")
            crop_on = st.checkbox("Crop before processing", value=False)
            if crop_on:
                left = st.slider("Left", 0, max(w - 1, 0), 0)
                top = st.slider("Top", 0, max(h - 1, 0), 0)
                right = st.slider("Right", 1, w, w)
                bottom = st.slider("Bottom", 1, h, h)
                work = _apply_crop(image, left, top, right, bottom)
                st.image(_bgr_to_rgb(work), use_container_width=True)
            else:
                work = image
                st.caption("Leave crop off to process the full photo.")
                st.image(_bgr_to_rgb(image), use_container_width=True)

    st.markdown("---")
    process = st.button("Process Diagram", type="primary", disabled=image is None)
    if image is None:
        st.info("Upload a photo or load the demo whiteboard to begin.")

    if process and work is not None:
        try:
            model, device = _cached_classifier()
            ocr = _cached_ocr()
            with st.spinner("Running localization, CNN, OCR, and graph matching..."):
                result = run_pipeline(work, model=model, device=device, ocr=ocr)
            st.session_state.graph = result.graph
            st.session_state.pipeline_log = result.log_lines
            st.session_state.status = (
                f"Done: {result.n_nodes} nodes, {result.n_edges} undirected edges."
            )
            st.success(st.session_state.status)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    st.markdown("---")
    diag_col, edit_col = st.columns([3, 2])
    with diag_col:
        st.subheader("Digital diagram")
        st.caption(st.session_state.status)
        st_draw_graph(st.session_state.graph)
        if st.session_state.pipeline_log:
            with st.expander("Pipeline log"):
                st.text("\n".join(st.session_state.pipeline_log))
    with edit_col:
        _editor_panel()


if __name__ == "__main__":
    main()
