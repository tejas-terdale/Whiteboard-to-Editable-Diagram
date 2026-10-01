# Whiteboard Photo to Editable Diagram

Takes a photo of a whiteboard diagram and reconstructs it as an editable digital graph (shapes, labels, undirected connections) in Streamlit.

Localization uses classical OpenCV contours. The deep learning piece is a MobileNetV2 classifier (`box`, `circle`, `diamond`, `arrow`) trained on synthetic crops. Text is read with EasyOCR on shape crops only. Connections are undirected.

## Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

1. Upload a PNG/JPG photo, or click **Load demo whiteboard**.
2. Optionally crop the photo, then click **Process Diagram**.
3. Rename, add, or delete nodes and undirected edges with the form widgets (no drag-and-drop canvas).
4. Download PNG or PDF from the sidebar.

The reconstructed NetworkX graph lives in `st.session_state['graph']` so edits survive Streamlit reruns.

## Setup

Python 3.9+ is required.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Graphviz must also be installed **as a system binary** (`dot` on PATH), not only the Python wheel:

- Windows: `winget install Graphviz.Graphviz` then restart the terminal
- macOS: `brew install graphviz`
- Linux: `sudo apt install graphviz` (or equivalent)

Train the classifier before processing photos (weights at `models/shape_classifier.pth`):

```bash
python scripts/generate_synthetic_data.py
python scripts/train_classifier.py
python scripts/verify_env.py
```

## Other commands

```bash
python scripts/evaluate_localization_dl.py
python scripts/test_phase3_pipeline.py
python scripts/test_phase3_pipeline.py --image path/to/photo.jpg
```

Real test photos can go in `data/test_real/`.

## Layout

```
├── app.py                       # Streamlit UI
├── project_context.txt
├── requirements.txt
├── README.md
├── data/
│   ├── synthetic/train|val/...
│   └── test_real/
├── models/shape_classifier.pth
├── src/
│   ├── localization.py
│   ├── model.py
│   ├── ocr.py
│   ├── relationships.py
│   ├── graph_builder.py
│   ├── pipeline.py
│   ├── renderer.py
│   └── export.py
└── scripts/
```

## Pipeline

1. OpenCV preprocess (grayscale + adaptive threshold; no automatic deskew).
2. Contour-based candidate localization (not YOLO).
3. Fine-tuned CNN classifies each crop: box / circle / diamond / arrow.
4. OCR on shape crops only (EasyOCR).
5. Geometry-only undirected edge matching.
6. NetworkX graph → Graphviz in Streamlit, form-widget editing, PNG/PDF export.
