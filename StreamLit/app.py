import os
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import torch
import open_clip

from ultralytics import YOLO
from PIL import Image, ImageEnhance

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VR_Project · Fashion Retrieval",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS  —  Luxury Editorial Dark Theme
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Josefin+Sans:wght@300;400;600&family=JetBrains+Mono:wght@300;400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body { background: #0C0C0E; }

.stApp {
    background: #0C0C0E;
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(196,168,128,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 90% 60%, rgba(196,168,128,0.03) 0%, transparent 60%);
}

[data-testid="stSidebar"] {
    background: #0E0E10 !important;
    border-right: 1px solid #1C1C20 !important;
}

[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.4rem; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding: 0 3rem 4rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hero ── */
.hero-wrap {
    padding: 3.5rem 0 2.5rem 0;
    border-bottom: 1px solid #1C1C20;
    margin-bottom: 2.5rem;
}

.hero-eyebrow {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #C4A880;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.hero-eyebrow::after {
    content: '';
    display: inline-block;
    width: 40px;
    height: 1px;
    background: #C4A880;
    opacity: 0.5;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.8rem;
    font-weight: 300;
    color: #F2EDE6;
    line-height: 0.95;
    letter-spacing: -1px;
}

.hero-title em { font-style: italic; color: #C4A880; }

.hero-subtitle {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6A6A7A;
    margin-top: 1rem;
}

/* ── Sidebar ── */
.sb-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 300;
    color: #F2EDE6;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1C1C20;
    padding-bottom: 1.2rem;
    margin-bottom: 0.4rem;
}

.sb-logo span { color: #C4A880; font-style: italic; }

.sb-section-title {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #C4A880;
    margin: 1.6rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sb-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2A2A30, transparent);
}

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid #16161A;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6A6A80;
    letter-spacing: 0.05em;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #C4A880;
}

.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #141418;
    border: 1px solid #222228;
    border-radius: 3px;
    padding: 4px 10px;
    margin: 3px 0;
    width: 100%;
}

.model-chip-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #C4A880;
    flex-shrink: 0;
}

.model-chip-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7A7A90;
    letter-spacing: 0.03em;
}

/* ── Step Row ── */
.step-row {
    display: flex;
    gap: 0;
    margin: 1.5rem 0;
}

.step-item {
    flex: 1;
    text-align: center;
    padding: 10px 6px;
    position: relative;
}

.step-item::after {
    content: '→';
    position: absolute;
    right: -6px;
    top: 50%;
    transform: translateY(-50%);
    color: #1E1E28;
    font-size: 0.7rem;
}

.step-item:last-child::after { display: none; }

.step-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 300;
    color: #C4A880;
    line-height: 1;
}

.step-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5A5A72;
    margin-top: 3px;
}

/* ── Panel ── */
.panel-title {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #C4A880;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1C1C22;
}

/* ── Badge Grid ── */
.badge-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.badge-item {
    background: #141418;
    border: 1px solid #1E1E26;
    border-radius: 3px;
    padding: 8px 12px;
}

.badge-key {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5A5A72;
    margin-bottom: 3px;
}

.badge-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #F2EDE6;
}

.badge-val.on  { color: #7ECFA8; }
.badge-val.off { color: #4A4A5A; }

/* ── Region Tag ── */
.region-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 2px;
    padding: 5px 12px;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.region-tag.upper { background: #12201A; border: 1px solid #1D3829; color: #7ECFA8; }
.region-tag.lower { background: #12161F; border: 1px solid #1D2740; color: #6EB0D4; }
.region-tag.full  { background: #1E1912; border: 1px solid #3A3018; color: #C4A880; }

/* ── Banners ── */
.banner {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    border-radius: 3px;
    padding: 10px 14px;
    margin: 10px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
}

.banner.info    { background:#0D1620; border:1px solid #192840; color:#7AAACC; border-left:2px solid #4A90C4; }
.banner.warn    { background:#1A1608; border:1px solid #342C10; color:#BF9A50; border-left:2px solid #C4A030; }
.banner.success { background:#0C1810; border:1px solid #182818; color:#6ABD90; border-left:2px solid #5ABD8A; }

/* ── Divider ── */
.divider-label {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 2rem 0;
}

.divider-label::before, .divider-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #181820;
}

.divider-label span {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #4E4E60;
    white-space: nowrap;
}

/* ── Result Cards ── */
.result-body {
    padding: 12px 14px 14px 14px;
    background: #0E0E10;
    border: 1px solid #1C1C22;
    border-top: none;
    border-radius: 0 0 4px 4px;
    margin-top: -4px;
}

.result-score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.result-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #C4A880;
}

.result-score-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5A5A72;
}

.sim-track {
    width: 100%;
    height: 2px;
    background: #181820;
    border-radius: 1px;
    margin-bottom: 10px;
    overflow: hidden;
}

.sim-fill {
    height: 2px;
    border-radius: 1px;
    background: linear-gradient(90deg, #7ECFA8, #C4A880);
}

.result-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #5A5A72;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.result-caption {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: #7A7A90;
    line-height: 1.5;
}

/* ── Results header ── */
.results-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 1.5rem;
}

.results-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 300;
    color: #F2EDE6;
    font-style: italic;
}

.results-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #5A5A72;
    letter-spacing: 0.08em;
}

/* ── Buttons ── */
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background: transparent !important;
    border: 1px solid #C4A880 !important;
    color: #C4A880 !important;
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2.5rem !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}

[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background: #C4A880 !important;
    color: #0C0C0E !important;
}

/* ── Widgets ── */
label, .stRadio label, .stCheckbox label {
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.80rem !important;
    letter-spacing: 0.1em !important;
    color: #7A7A90 !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: #C4A880 !important;
    border-color: #C4A880 !important;
}

/* ── Footer ── */
.footer-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3A3A4A;
    letter-spacing: 0.15em;
    text-align: center;
    padding-top: 2rem;
    border-top: 1px solid #141418;
    margin-top: 3rem;
}
            section[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
}

/* Sidebar toggle button */
button[kind="header"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Top-right toolbar */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 1rem !important;
    left: 1rem !important;
    z-index: 999999 !important;
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    padding: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="sb-logo">VR_<span>PROJECT</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Results</div>', unsafe_allow_html=True)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, label_visibility="collapsed")
    st.markdown(
        f'<div style="text-align:right;font-family:JetBrains Mono,monospace;'
        f'font-size:0.7rem;color:#C4A880;margin-top:2px">K = {top_k}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-section-title">Crop Strategy</div>', unsafe_allow_html=True)
    query_mode = st.radio(
        "Mode",
        options=["Auto-Detect Region", "Full Body", "Upper Body", "Lower Body", "Manual Upload"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="sb-section-title">Embedding</div>', unsafe_allow_html=True)
    use_multicrop = st.checkbox("Multi-Crop Fusion", value=True,
        help="Encodes 3 crops and averages. Reduces pose/background noise.")

    st.markdown('<div class="sb-section-title">Retrieval</div>', unsafe_allow_html=True)
    filter_by_category = st.checkbox("Category Filter", value=True,
        help="Restricts FAISS to gallery items matching the detected clothing region.")
    use_reranking = st.checkbox("Cosine Reranking", value=True,
        help="Exact cosine reranking of HNSW candidates.")
    dedup_results = st.checkbox("Deduplicate Items", value=True,
        help="Removes duplicate item_id entries.")

    st.markdown('<div class="sb-section-title">Stack</div>', unsafe_allow_html=True)
    for chip in [
        "YOLOv8n  ·  Detection",
        "CLIP ViT-B/32  ·  Encoder",
        "HNSW FAISS  ·  ANN",
        "DeepFashion  ·  Dataset"
    ]:
        st.markdown(f"""
        <div class="model-chip">
            <div class="model-chip-dot"></div>
            <span class="model-chip-text">{chip}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Eval Metrics</div>', unsafe_allow_html=True)
    for label, val in [
        ("Recall@5","41.0%"), ("Recall@10","46.4%"), ("Recall@15","49.1%"),
        ("mAP@5","11.6%"),    ("mAP@10","12.3%"),    ("mAP@15","12.6%"),
        ("NDCG@5","18.8%"),   ("NDCG@10","18.7%"),   ("NDCG@15","19.2%"),
    ]:
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">{label}</span>
            <span class="metric-value">{val}</span>
        </div>""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">✦  Visual Retrieval System</div>
    <div class="hero-title">Visual <em>Fashion</em><br>Retrieval</div>
    <div class="hero-subtitle">Upload · Detect · Encode · Retrieve · Explore</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="step-row">
    <div class="step-item"><div class="step-num">01</div><div class="step-label">Upload</div></div>
    <div class="step-item"><div class="step-num">02</div><div class="step-label">YOLO Detect</div></div>
    <div class="step-item"><div class="step-num">03</div><div class="step-label">Crop Region</div></div>
    <div class="step-item"><div class="step-num">04</div><div class="step-label">CLIP Encode</div></div>
    <div class="step-item"><div class="step-num">05</div><div class="step-label">HNSW Search</div></div>
    <div class="step-item"><div class="step-num">06</div><div class="step-label">Results</div></div>
</div>
<div class="divider-label"><span>Upload Query</span></div>
""", unsafe_allow_html=True)

# =========================================================
# MODELS + DATA
# =========================================================

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess, device

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

@st.cache_data
def load_data():
    embs = np.load("embeddings.npy").astype("float32")
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    meta = pd.read_csv("metadata_with_embeddings.csv")
    gdf = meta[meta["split"] == "gallery"].reset_index(drop=True)
    gembs = embs[gdf["embedding_index"].values].astype("float32")
    return embs, gdf, gembs

@st.cache_resource
def build_faiss_index(gembs):
    idx = faiss.IndexHNSWFlat(gembs.shape[1], 32)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    idx.add(gembs)
    return idx

@st.cache_data
def build_category_subsets(gdf):
    subsets = {}
    if "crop_class" in gdf.columns:
        for cat in gdf["crop_class"].unique():
            subsets[cat] = gdf[gdf["crop_class"] == cat].index.tolist()
    return subsets

model, preprocess, device = load_clip_model()
yolo_model = load_yolo_model()
embeddings, gallery_df, gallery_embeddings = load_data()
gallery_index = build_faiss_index(gallery_embeddings)
category_subsets = build_category_subsets(gallery_df)

# =========================================================
# LOGIC
# =========================================================

def encode_image(image, use_fusion=True):
    def _enc(img):
        t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(t).cpu().numpy().astype("float32")
        f /= np.linalg.norm(f) + 1e-8
        return f

    if not use_fusion:
        return _enc(image)

    w, h = image.size
    mw, mh = int(w * 0.05), int(h * 0.05)
    crops = [
        image,
        image.crop((mw, mh, w - mw, h - mh)),
        ImageEnhance.Brightness(image).enhance(1.15)
    ]
    fused = np.mean([_enc(c) for c in crops], axis=0)
    fused /= np.linalg.norm(fused) + 1e-8
    return fused.astype("float32")


def classify_crop_region(box, image_height):
    _, y1, _, y2 = box
    tf, bf = y1 / image_height, y2 / image_height
    if tf < 0.1 and bf > 0.75: return "full"
    if bf < 0.55: return "upper"
    if tf > 0.4:  return "lower"
    return "full"


def detect_and_crop(image, mode):
    if mode == "Manual Upload":
        return image, "full", None

    results = yolo_model(image)
    boxes = results[0].boxes
    largest_box, largest_area = None, 0

    if boxes is not None:
        for box in boxes:
            if int(box.cls[0]) != 0: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = [x1, y1, x2, y2]

    if largest_box is None:
        return image, "full", None

    x1, y1, x2, y2 = map(int, largest_box)
    person_crop = image.crop((x1, y1, x2, y2))
    pw, ph = person_crop.size
    auto_region = classify_crop_region(largest_box, image.height)

    region = {
        "Auto-Detect Region": auto_region,
        "Upper Body": "upper",
        "Lower Body": "lower",
        "Full Body": "full"
    }.get(mode, auto_region)

    if region == "upper":
        crop = person_crop.crop((0, 0, pw, int(ph * 0.55)))
    elif region == "lower":
        crop = person_crop.crop((0, int(ph * 0.45), pw, ph))
    else:
        crop = person_crop

    return crop, region, auto_region


def get_category_filter(region, subsets):
    upper_cats = {"upper","top","shirt","jacket","blouse","coat","sweater","hoodie"}
    lower_cats = {"lower","bottom","pants","jeans","skirt","shorts","trousers"}
    target = upper_cats if region == "upper" else lower_cats if region == "lower" else None
    if target is None: return None
    matched = []
    for cat, idxs in subsets.items():
        if any(k in cat.lower() for k in target):
            matched.extend(idxs)
    return matched if matched else None


def retrieve(qemb, top_k, cat_filter, use_reranking, dedup):
    oversample = top_k * 8

    if cat_filter and len(cat_filter) > 0:
        sub_embs = gallery_embeddings[cat_filter].astype("float32")
        sub_idx = faiss.IndexFlatIP(sub_embs.shape[1])
        sub_idx.add(sub_embs)
        scores, local_idx = sub_idx.search(qemb, min(oversample, len(cat_filter)))
        gidxs = [cat_filter[i] for i in local_idx[0] if i < len(cat_filter)]
        raw_scores = scores[0][:len(gidxs)]
    else:
        scores, idxs = gallery_index.search(qemb, oversample)
        gidxs = idxs[0].tolist()
        raw_scores = scores[0]

    candidates = [
        {"gallery_idx": idx, "score": float(raw_scores[i])}
        for i, idx in enumerate(gidxs)
        if 0 <= idx < len(gallery_df)
    ]

    if use_reranking and candidates:
        cidxs = [c["gallery_idx"] for c in candidates]
        cembs = gallery_embeddings[cidxs]
        exact = (cembs @ qemb.T).squeeze()
        for i, c in enumerate(candidates):
            c["score"] = float(exact[i]) if hasattr(exact, "__len__") else float(exact)
        candidates.sort(key=lambda x: x["score"], reverse=True)

    if dedup:
        seen, out = set(), []
        for c in candidates:
            iid = gallery_df.iloc[c["gallery_idx"]].get("item_id", c["gallery_idx"])
            if iid not in seen:
                seen.add(iid)
                out.append(c)
        candidates = out

    return candidates[:top_k * 3]

# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Drop a fashion image here  ·  JPG / PNG / WEBP",
    type=["jpg", "jpeg", "png", "webp"]
)

# =========================================================
# MAIN
# =========================================================

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    col_q, col_p = st.columns([1, 2], gap="large")

    with col_q:
        st.markdown('<div class="panel-title">Query Image</div>', unsafe_allow_html=True)
        st.image(query_image, use_container_width=True)

    with col_p:
        st.markdown('<div class="panel-title">Active Pipeline</div>', unsafe_allow_html=True)

        def _cls(flag): return "on" if flag else "off"
        def _val(flag): return "ON" if flag else "OFF"

        st.markdown(f"""
        <div class="badge-grid">
            <div class="badge-item">
                <div class="badge-key">Crop Mode</div>
                <div class="badge-val">{query_mode.replace("Auto-Detect Region","Auto")}</div>
            </div>
            <div class="badge-item">
                <div class="badge-key">Top K</div>
                <div class="badge-val">{top_k}</div>
            </div>
            <div class="badge-item">
                <div class="badge-key">Multi-Crop</div>
                <div class="badge-val {_cls(use_multicrop)}">{_val(use_multicrop)}</div>
            </div>
            <div class="badge-item">
                <div class="badge-key">Cat. Filter</div>
                <div class="badge-val {_cls(filter_by_category)}">{_val(filter_by_category)}</div>
            </div>
            <div class="badge-item">
                <div class="badge-key">Reranking</div>
                <div class="badge-val {_cls(use_reranking)}">{_val(use_reranking)}</div>
            </div>
            <div class="badge-item">
                <div class="badge-key">Dedup</div>
                <div class="badge-val {_cls(dedup_results)}">{_val(dedup_results)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider-label"><span>Region Detection</span></div>', unsafe_allow_html=True)

    cropped_image, detected_region, auto_region = detect_and_crop(query_image, mode=query_mode)
    region_icon = {"upper": "↑", "lower": "↓", "full": "↕"}.get(detected_region, "↕")

    col_c1, col_c2 = st.columns([1, 2], gap="large")

    with col_c1:
        st.markdown(
            f'<div class="panel-title">Crop &nbsp;'
            f'<span class="region-tag {detected_region}">{region_icon} {detected_region.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.image(cropped_image, use_container_width=True)

    with col_c2:
        st.markdown('<div class="panel-title">Detection Info</div>', unsafe_allow_html=True)

        if auto_region:
            st.markdown(f"""
            <div class="banner info">
                YOLO auto-detected: <strong>{auto_region.upper()}</strong> &nbsp;·&nbsp;
                Applied crop mode: <strong>{detected_region.upper()}</strong>
            </div>
            <div class="banner info">
                Override via <strong>Crop Strategy</strong> in sidebar.<br>
                Upper Body → top 55% of person crop.<br>
                Lower Body → bottom 55% of person crop.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="banner warn">
                No person detected by YOLO · using original image as query.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    retrieve_btn = st.button("✦  RETRIEVE SIMILAR PRODUCTS")

    if retrieve_btn:
        with st.spinner("Encoding · Searching · Reranking…"):
            qemb = encode_image(cropped_image, use_fusion=use_multicrop)
            cat_filter = get_category_filter(detected_region, category_subsets) if filter_by_category else None
            candidates = retrieve(
                qemb.reshape(1, -1),
                top_k=top_k,
                cat_filter=cat_filter,
                use_reranking=use_reranking,
                dedup=dedup_results
            )

        filter_note = f"  ·  filtered to {len(cat_filter)} items" if cat_filter else ""
        st.markdown(f"""
        <div class="banner success">
            ✓ &nbsp; Retrieved {len(candidates)} candidates · showing top {top_k}{filter_note}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="divider-label"><span>Retrieved Results</span></div>
        <div class="results-header">
            <div class="results-title">Similar Products</div>
            <div class="results-count">top {top_k} &nbsp;·&nbsp; {detected_region} region &nbsp;·&nbsp; cosine similarity</div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(top_k, gap="small")
        valid_results = 0

        for cand in candidates:
            if valid_results >= top_k:
                break

            idx = cand["gallery_idx"]
            similarity = cand["score"]

            try:
                row = gallery_df.iloc[idx]
                filename = row["image_path"].split("/")[-1]
                crop_class = row["crop_class"]
                img_path = os.path.join("cropped_products", crop_class, filename)

                if not os.path.exists(img_path):
                    continue

                item_id = row["item_id"]
                caption = row.get("caption", "") or ""
                retrieved_img = Image.open(img_path)
                sim_pct = int(min(max(float(similarity), 0.0), 1.0) * 100)

                with cols[valid_results]:
                    st.image(retrieved_img, use_container_width=True)
                    st.markdown(f"""
                    <div class="result-body">
                        <div class="result-score-row">
                            <div>
                                <div class="result-score-label">Similarity</div>
                                <div class="result-score-num">{similarity:.4f}</div>
                            </div>
                            <span class="region-tag {detected_region}" style="font-size:0.55rem;padding:3px 8px">
                                #{valid_results + 1}
                            </span>
                        </div>
                        <div class="sim-track">
                            <div class="sim-fill" style="width:{sim_pct}%"></div>
                        </div>
                        <div class="result-id">{item_id} &nbsp;·&nbsp; {crop_class}</div>
                        <div class="result-caption">{caption[:100] if caption else "—"}</div>
                    </div>
                    """, unsafe_allow_html=True)

                valid_results += 1

            except Exception:
                continue

        if valid_results == 0:
            st.markdown("""
            <div class="banner warn">
                No images found. Try disabling Category Filter or switching to Full Body mode.
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer-text">
    VR_Project &nbsp;·&nbsp; YOLO &nbsp;·&nbsp; CLIP ViT-B/32 &nbsp;·&nbsp;
    HNSW FAISS &nbsp;·&nbsp; MULTI-CROP FUSION &nbsp;·&nbsp; COSINE RERANKING
</div>
""", unsafe_allow_html=True)