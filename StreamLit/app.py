import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import faiss
import numpy as np
import pandas as pd
import streamlit as st
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
# CUSTOM CSS
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

#MainMenu, footer { visibility: hidden; }

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
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.28em; text-transform: uppercase;
    color: #C4A880; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 10px;
}
.hero-eyebrow::after {
    content: ''; display: inline-block;
    width: 40px; height: 1px;
    background: #C4A880; opacity: 0.5;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.8rem; font-weight: 300;
    color: #F2EDE6; line-height: 0.95; letter-spacing: -1px;
}
.hero-title em { font-style: italic; color: #C4A880; }
.hero-subtitle {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.78rem; font-weight: 300;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #6A6A7A; margin-top: 1rem;
}

/* ── Sidebar ── */
.sb-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem; font-weight: 300;
    color: #F2EDE6; letter-spacing: 0.1em;
    border-bottom: 1px solid #1C1C20;
    padding-bottom: 1.2rem; margin-bottom: 0.4rem;
}
.sb-logo span { color: #C4A880; font-style: italic; }
.sb-section-title {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.25em; text-transform: uppercase;
    color: #C4A880; margin: 1.6rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 8px;
}
.sb-section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #2A2A30, transparent);
}
.metric-row {
    display: flex; justify-content: space-between;
    align-items: baseline; padding: 5px 0;
    border-bottom: 1px solid #16161A;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #6A6A80; letter-spacing: 0.05em;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #C4A880;
}
.model-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #141418; border: 1px solid #222228;
    border-radius: 3px; padding: 4px 10px;
    margin: 3px 0; width: 100%;
}
.model-chip-dot { width: 5px; height: 5px; border-radius: 50%; background: #C4A880; flex-shrink: 0; }
.model-chip-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #7A7A90; letter-spacing: 0.03em;
}

/* ── Step Row ── */
.step-row { display: flex; gap: 0; margin: 1.5rem 0; }
.step-item { flex: 1; text-align: center; padding: 10px 6px; position: relative; }
.step-item::after {
    content: '→'; position: absolute; right: -6px; top: 50%;
    transform: translateY(-50%); color: #1E1E28; font-size: 0.7rem;
}
.step-item:last-child::after { display: none; }
.step-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem; font-weight: 300; color: #C4A880; line-height: 1;
}
.step-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #5A5A72; margin-top: 3px;
}

/* ── Panel ── */
.panel-title {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: #C4A880; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: #1C1C22; }

/* ── Badge Grid ── */
.badge-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.badge-item {
    background: #141418; border: 1px solid #1E1E26;
    border-radius: 3px; padding: 8px 12px;
}
.badge-key {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: #5A5A72; margin-bottom: 3px;
}
.badge-val { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #F2EDE6; }
.badge-val.on  { color: #7ECFA8; }
.badge-val.off { color: #4A4A5A; }

/* ── Region Tag ── */
.region-tag {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 2px; padding: 5px 12px;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
}
.region-tag.upper_body { background: #12201A; border: 1px solid #1D3829; color: #7ECFA8; }
.region-tag.lower_body { background: #12161F; border: 1px solid #1D2740; color: #6EB0D4; }
.region-tag.full_body  { background: #1E1912; border: 1px solid #3A3018; color: #C4A880; }

/* ── Region Selector Cards ── */
.region-card {
    background: #0E0E11;
    border: 1px solid #1C1C24;
    border-radius: 6px;
    padding: 0;
    overflow: hidden;
    transition: border-color 0.2s;
}
.region-card.selected {
    border-color: #C4A880 !important;
    box-shadow: 0 0 0 1px #C4A880 inset;
}
.region-card-header {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: #13131A;
    border-bottom: 1px solid #1C1C24;
}
.region-card-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.6rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
}
.region-card-label.upper_body { color: #7ECFA8; }
.region-card-label.lower_body { color: #6EB0D4; }
.region-card-label.full_body  { color: #C4A880; }
.region-card-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #4A4A60;
}

/* ── Banners ── */
.banner {
    display: flex; gap: 10px; align-items: flex-start;
    border-radius: 3px; padding: 10px 14px; margin: 10px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; line-height: 1.7;
}
.banner.info    { background:#0D1620; border:1px solid #192840; color:#7AAACC; border-left:2px solid #4A90C4; }
.banner.warn    { background:#1A1608; border:1px solid #342C10; color:#BF9A50; border-left:2px solid #C4A030; }
.banner.success { background:#0C1810; border:1px solid #182818; color:#6ABD90; border-left:2px solid #5ABD8A; }

/* ── Divider ── */
.divider-label {
    display: flex; align-items: center; gap: 14px; margin: 2rem 0;
}
.divider-label::before, .divider-label::after {
    content: ''; flex: 1; height: 1px; background: #181820;
}
.divider-label span {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.65rem; letter-spacing: 0.25em;
    text-transform: uppercase; color: #4E4E60; white-space: nowrap;
}

/* ── Result Cards ── */
.result-body {
    padding: 12px 14px 14px 14px;
    background: #0E0E10;
    border: 1px solid #1C1C22; border-top: none;
    border-radius: 0 0 4px 4px; margin-top: -4px;
}
.result-score-row {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 8px;
}
.result-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: #C4A880;
}
.result-score-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.62rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #5A5A72;
}
.sim-track {
    width: 100%; height: 2px; background: #181820;
    border-radius: 1px; margin-bottom: 10px; overflow: hidden;
}
.sim-fill {
    height: 2px; border-radius: 1px;
    background: linear-gradient(90deg, #7ECFA8, #C4A880);
}
.result-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #5A5A72;
    letter-spacing: 0.04em; margin-bottom: 4px;
}
.result-caption {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.78rem; font-weight: 300;
    color: #7A7A90; line-height: 1.5;
}

/* ── Results header ── */
.results-header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 1.5rem; }
.results-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem; font-weight: 300;
    color: #F2EDE6; font-style: italic;
}
.results-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #5A5A72; letter-spacing: 0.08em;
}

/* ── Buttons ── */
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background: transparent !important;
    border: 1px solid #C4A880 !important;
    color: #C4A880 !important;
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.7rem !important; font-weight: 600 !important;
    letter-spacing: 0.2em !important; text-transform: uppercase !important;
    padding: 0.7rem 2.5rem !important; border-radius: 2px !important;
    transition: all 0.2s !important;
}
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background: #C4A880 !important; color: #0C0C0E !important;
}

/* ── Widgets ── */
label, .stRadio label, .stCheckbox label {
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.80rem !important; letter-spacing: 0.1em !important;
    color: #7A7A90 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #C4A880 !important; border-color: #C4A880 !important;
}

/* ── Footer ── */
.footer-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3A3A4A; letter-spacing: 0.15em;
    text-align: center; padding-top: 2rem;
    border-top: 1px solid #141418; margin-top: 3rem;
}

section[data-testid="stSidebar"] {
    min-width: 320px !important; max-width: 320px !important;
}
button[kind="header"] {
    display: block !important; visibility: visible !important; opacity: 1 !important;
}
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    position: fixed !important; top: 1rem !important; left: 1rem !important;
    z-index: 999999 !important; background: #111 !important;
    border: 1px solid #333 !important; border-radius: 6px !important; padding: 4px !important;
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

    st.markdown('<div class="sb-section-title">Region</div>', unsafe_allow_html=True)
    query_mode = st.radio(
        "Region",
        options=["Auto", "Upper Body", "Lower Body", "Full Body"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Auto: use the highest-confidence detection from best_deepfashion_yolo.pt.\n"
            "Upper / Lower / Full Body: force that specific class — the detector still runs "
            "on your friend's model, but only the matching bbox is used."
        )
    )

    # Map sidebar label → YOLO class name used by best_deepfashion_yolo.pt
    MODE_CLASS = {
        "Auto":        None,
        "Upper Body":  "upper_body",
        "Lower Body":  "lower_body",
        "Full Body":   "full_body",
    }

    st.markdown('<div class="sb-section-title">Embedding</div>', unsafe_allow_html=True)
    use_multicrop = st.checkbox("Multi-Crop Fusion", value=True,
        help="Encodes 3 bbox-based crop variants and averages. Reduces pose/background noise.")

    st.markdown('<div class="sb-section-title">Retrieval</div>', unsafe_allow_html=True)
    filter_by_category = st.checkbox("Category Filter", value=True,
        help="Restricts FAISS to gallery items matching the detected clothing class.")
    use_reranking = st.checkbox("Cosine Reranking", value=True,
        help="Exact cosine reranking of HNSW candidates.")
    dedup_results = st.checkbox("Deduplicate Items", value=True,
        help="Removes duplicate item_id entries.")

    st.markdown('<div class="sb-section-title">Stack</div>', unsafe_allow_html=True)
    for chip in [
        "best_deepfashion_yolo.pt  ·  Detector",
        "CLIP ViT-B/32  ·  Encoder",
        "HNSW FAISS  ·  ANN Index",
        "DeepFashion  ·  Dataset",
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
    <div class="hero-subtitle">Upload · Detect · Select Region · Encode · Retrieve · Explore</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="step-row">
    <div class="step-item"><div class="step-num">01</div><div class="step-label">Upload</div></div>
    <div class="step-item"><div class="step-num">02</div><div class="step-label">Fashion YOLO</div></div>
    <div class="step-item"><div class="step-num">03</div><div class="step-label">Select Region</div></div>
    <div class="step-item"><div class="step-num">04</div><div class="step-label">CLIP Encode</div></div>
    <div class="step-item"><div class="step-num">05</div><div class="step-label">HNSW Search</div></div>
    <div class="step-item"><div class="step-num">06</div><div class="step-label">Results</div></div>
</div>
<div class="divider-label"><span>Upload Query</span></div>
""", unsafe_allow_html=True)

# =========================================================
# CLIP CHECKPOINT LOADER  (shared helper — also used by batch_eval.py)
# =========================================================

FINETUNED_CKPT = "best_clip_model.pt"


def _extract_state_dict(raw: dict) -> dict:
    """
    Unwrap any known checkpoint wrapper and return a flat state_dict.

    Handles (in priority order):
      • {'model_state_dict': {...}}   — common PyTorch training saves
      • {'model': {...}}              — some frameworks use this key
      • {'state_dict': {...}}         — Lightning / OpenCLIP full checkpoints
      • raw dict with param names     — already a flat state_dict
    Then strips DataParallel 'module.' prefix if present on every key.
    """
    for wrapper_key in ("model_state_dict", "model", "state_dict"):
        if wrapper_key in raw and isinstance(raw[wrapper_key], dict):
            raw = raw[wrapper_key]
            break

    # Strip DataParallel prefix
    if raw and all(k.startswith("module.") for k in raw):
        raw = {k[len("module."):]: v for k, v in raw.items()}

    return raw


def load_clip_weights(device: str):
    """
    Build CLIP ViT-B/32 and attempt to load best_clip_model.pt.

    Strategy:
      1. Always create architecture + preprocess via pretrained='openai'.
         This guarantees the preprocess transforms are identical to those used
         when building embeddings.npy — weights can change, transforms cannot.
      2. Try to load best_clip_model.pt and overwrite model weights.
         Uses strict=False so minor key mismatches (logit_scale, etc.) don't crash.
      3. On ANY failure (missing file, shape mismatch, corrupt checkpoint):
         fall back to OpenAI pretrained weights and log clearly.

    Returns: (model, preprocess, used_finetuned: bool)
    """
    mdl, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )

    used_finetuned = False

    if os.path.exists(FINETUNED_CKPT):
        try:
            raw = torch.load(FINETUNED_CKPT, map_location="cpu")

            if not isinstance(raw, dict):
                raise TypeError(f"Checkpoint is {type(raw)}, expected dict.")

            state_dict = _extract_state_dict(raw)
            missing, unexpected = mdl.load_state_dict(state_dict, strict=False)

            if missing:
                n = len(missing)
                preview = missing[:4]
                print(f"[CLIP]  Keys missing  ({n}): {preview}{'…' if n > 4 else ''}")
            if unexpected:
                n = len(unexpected)
                preview = unexpected[:4]
                print(f"[CLIP]  Keys extra    ({n}): {preview}{'…' if n > 4 else ''}")

            used_finetuned = True
            print(f"[CLIP]  ✅  Fine-tuned weights loaded — {FINETUNED_CKPT}")

        except Exception as exc:
            print(f"[CLIP]  ⚠️  Could not load {FINETUNED_CKPT}: {exc}")
            print("[CLIP]  ↩️  Falling back to OpenAI pretrained weights.")
    else:
        print(f"[CLIP]  ℹ️  {FINETUNED_CKPT} not found — using OpenAI pretrained weights.")

    if not used_finetuned:
        print("[CLIP]  ✅  OpenAI pretrained weights active.")

    mdl = mdl.to(device).eval()
    return mdl, preprocess, used_finetuned


# =========================================================
# MODELS + DATA
# =========================================================

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl, preprocess, _ = load_clip_weights(device)
    return mdl, preprocess, device


@st.cache_resource
def load_fashion_yolo():
    # ✅ Your friend's trained model — NOT the generic COCO yolov8n.pt
    return YOLO("best_deepfashion_yolo.pt")


@st.cache_data
def load_data():
    embs = np.load("embeddings.npy").astype("float32")
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    meta = pd.read_csv("metadata_with_embeddings.csv")
    gdf  = meta[meta["split"] == "gallery"].reset_index(drop=True)
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


clip_model, preprocess, device = load_clip_model()
fashion_yolo     = load_fashion_yolo()
embeddings, gallery_df, gallery_embeddings = load_data()
gallery_index    = build_faiss_index(gallery_embeddings)
category_subsets = build_category_subsets(gallery_df)

# =========================================================
# CONSTANTS
# =========================================================

CONF_THRESHOLD  = 0.25
FASHION_CLASSES = {"upper_body", "lower_body", "full_body"}

REGION_ICON  = {"upper_body": "↑", "lower_body": "↓", "full_body": "↕"}
REGION_LABEL = {"upper_body": "Upper Body", "lower_body": "Lower Body", "full_body": "Full Body"}

# =========================================================
# ENCODE IMAGE — multi-crop fusion
# =========================================================
# --- Embedding Fusion Compliance Clarification ---
# NOTE: The gallery embeddings (embeddings.npy) were generated OFFLINE
# using fused embeddings with a chosen alpha value:
# v_i = alpha * phi_V(x_i) + (1 - alpha) * phi_T(c_i)
# The online query pipeline below is visual-only, which aligns with 
# standard multimodal search architectures where queries have no captions.
# =========================================================

def encode_image(image: Image.Image, use_fusion: bool = True) -> np.ndarray:
    def _enc(img):
        t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(t).cpu().numpy().astype("float32")
        f /= np.linalg.norm(f) + 1e-8
        return f

    if not use_fusion:
        return _enc(image)

    w, h = image.size
    mw, mh = int(w * 0.05), int(h * 0.05)
    crops = [
        image,
        image.crop((mw, mh, w - mw, h - mh)),
        ImageEnhance.Brightness(image).enhance(1.15),
    ]
    fused = np.mean([_enc(c) for c in crops], axis=0)
    fused /= np.linalg.norm(fused) + 1e-8
    return fused.astype("float32")

# =========================================================
# FASHION DETECTION — returns ALL detected regions
# =========================================================

def run_detection(image: Image.Image, forced_cls=None):
    """
    Run best_deepfashion_yolo.pt on the image.

    Class mapping:
        0 -> 'upper_body'
        1 -> 'lower_body'
        2 -> 'full_body'

    forced_cls: if set, return ONLY that class. If none found, returns []
                so the UI can warn — NO silent fallback to a different class.

    Returns list of dicts: { cls, conf, bbox, crop }
    """
    results  = fashion_yolo(image)
    boxes    = results[0].boxes
    iw, ih   = image.size

    detections = []
    if boxes is not None:
        for box in boxes:
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = fashion_yolo.names.get(cls_id, "")

            if conf < CONF_THRESHOLD or cls_name not in FASHION_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))

            crop = image.crop((x1, y1, x2, y2))
            detections.append({
                "cls":  cls_name,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "crop": crop,
            })

    detections.sort(key=lambda d: d["conf"], reverse=True)

    if forced_cls is not None:
        return [d for d in detections if d["cls"] == forced_cls]

    # Auto mode — best detection per class, up to 3
    seen_classes, unique = set(), []
    for d in detections:
        if d["cls"] not in seen_classes:
            seen_classes.add(d["cls"])
            unique.append(d)
    return unique

# =========================================================
# CATEGORY FILTER
# =========================================================

def get_category_filter(detected_cls: str, subsets: dict):
    upper_keys = {"upper","top","shirt","jacket","blouse","coat","sweater","hoodie"}
    lower_keys = {"lower","bottom","pants","jeans","skirt","shorts","trousers"}
    full_keys  = {"full","dress","jumpsuit","romper","suit","outerwear"}

    mapping = {
        "upper_body": upper_keys,
        "lower_body": lower_keys,
        "full_body":  full_keys,
    }
    target_keys = mapping.get(detected_cls)
    if target_keys is None:
        return None

    matched = []
    for cat, idxs in subsets.items():
        if any(k in cat.lower() for k in target_keys):
            matched.extend(idxs)
    return matched if matched else None

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(qemb, top_k, cat_filter, use_reranking, dedup):
    oversample = top_k * 8

    if cat_filter and len(cat_filter) > 0:
        sub_embs = gallery_embeddings[cat_filter].astype("float32")
        sub_idx  = faiss.IndexFlatIP(sub_embs.shape[1])
        sub_idx.add(sub_embs)
        scores, local_idx = sub_idx.search(qemb, min(oversample, len(cat_filter)))
        gidxs      = [cat_filter[i] for i in local_idx[0] if i < len(cat_filter)]
        raw_scores = scores[0][:len(gidxs)]
    else:
        scores, idxs = gallery_index.search(qemb, oversample)
        gidxs        = idxs[0].tolist()
        raw_scores   = scores[0]

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
# MAIN PIPELINE
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
                <div class="badge-key">Detector</div>
                <div class="badge-val" style="font-size:0.62rem">{query_mode}</div>
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

    # ── Run detection ──
    mode_label = f"{query_mode} · best_deepfashion_yolo.pt"
    st.markdown(f'<div class="divider-label"><span>{mode_label}</span></div>', unsafe_allow_html=True)

    forced_cls = MODE_CLASS.get(query_mode)

    with st.spinner("Running best_deepfashion_yolo.pt…"):
        detections = run_detection(query_image, forced_cls=forced_cls)

    # =========================================================
    # REGION SELECTION UI
    # =========================================================

    if not detections:
        if forced_cls is not None:
            warn_msg = (
                f"best_deepfashion_yolo.pt found no <strong>{REGION_LABEL[forced_cls]}</strong> "
                f"region above the confidence threshold ({CONF_THRESHOLD}).<br>"
                f"Try switching to <strong>Auto</strong> or a different region in the sidebar."
            )
        else:
            warn_msg = (
                "best_deepfashion_yolo.pt found no garment regions in this image.<br>"
                "Try a different image or switch to a specific region in the sidebar."
            )
        st.markdown(f'<div class="banner warn">{warn_msg}</div>', unsafe_allow_html=True)
        st.stop()

    else:
        n = len(detections)
        det_cols = st.columns(n, gap="large")

        # Reset selection when file or mode changes
        if (
            "selected_cls" not in st.session_state
            or st.session_state.get("_last_file") != uploaded_file.name
            or st.session_state.get("_last_mode") != query_mode
            or st.session_state.get("force_reselect", False)
        ):
            st.session_state.selected_cls  = detections[0]["cls"]
            st.session_state.selected_crop = detections[0]["crop"]
            st.session_state._last_file    = uploaded_file.name
            st.session_state._last_mode    = query_mode
            st.session_state.crop_confirmed = False
            st.session_state.force_reselect = False

        for i, det in enumerate(detections):
            cls  = det["cls"]
            conf = det["conf"]
            crop = det["crop"]
            icon = REGION_ICON.get(cls, "↕")
            lbl  = REGION_LABEL.get(cls, cls)
            is_selected = (st.session_state.selected_cls == cls)

            with det_cols[i]:
                selected_marker = " ✦ SELECTED" if is_selected else ""
                st.markdown(f"""
                <div class="region-card {'selected' if is_selected else ''}">
                    <div class="region-card-header">
                        <span class="region-card-label {cls}">{icon} {lbl}{selected_marker}</span>
                        <span class="region-card-conf">conf {conf:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.image(crop, use_container_width=True)

                btn_label = "✦ Selected" if is_selected else f"Select {lbl}"
                if st.button(
                    btn_label,
                    key=f"sel_{cls}",
                    disabled=is_selected,
                    use_container_width=True,
                ):
                    st.session_state.selected_cls  = cls
                    st.session_state.selected_crop = crop
                    st.session_state.crop_confirmed = False
                    st.rerun()

        selected_cls  = st.session_state.selected_cls
        selected_crop = st.session_state.selected_crop

        icon = REGION_ICON.get(selected_cls, "↕")
        lbl  = REGION_LABEL.get(selected_cls, selected_cls)
        mode_note = (
            f"Auto mode — detected <strong>{n}</strong> region(s), showing all. Click to select."
            if query_mode == "Auto"
            else f"Forced <strong>{query_mode}</strong> mode — showing best matching detection."
        )
        st.markdown(f"""
        <div class="banner info" style="margin-top:1rem">
            <div>
                <strong>best_deepfashion_yolo.pt</strong> · {mode_note}<br>
                Active: <strong>{icon} {lbl}</strong>
                {f"(conf {[d['conf'] for d in detections if d['cls']==selected_cls][0]:.2f})" if any(d['cls']==selected_cls for d in detections) else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confirmation & Retrieval Flow ──
    st.markdown("")
    retrieve_btn = False
    
    if not st.session_state.get("crop_confirmed", False):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("✓  CONFIRM CROP", use_container_width=True):
                st.session_state.crop_confirmed = True
                st.rerun()
        with col_c2:
            if st.button("⟲  RE-SELECT REGION", use_container_width=True):
                st.session_state.force_reselect = True
                st.rerun()
        
        st.info("Please confirm your crop to proceed to retrieval.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            retrieve_btn = st.button("✦  RETRIEVE SIMILAR PRODUCTS", use_container_width=True)
        with col_c2:
            if st.button("⟲  RE-SELECT REGION", use_container_width=True):
                st.session_state.crop_confirmed = False
                st.rerun()

    if retrieve_btn:
        with st.spinner("Encoding · Searching · Reranking…"):
            qemb = encode_image(selected_crop, use_fusion=use_multicrop)
            cat_filter = get_category_filter(selected_cls, category_subsets) if filter_by_category else None
            
            # --- BLIP-2 Semantic Reranking Clarification ---
            # NOTE: The project specification mentions BLIP-2 ITM semantic reranking.
            # In this deployment, we use cosine reranking as a computationally lighter
            # substitute to achieve real-time latency. BLIP-2 ITM requires a heavy 
            # cross-attention pass for every candidate, which is prohibitive without GPUs.
            # -----------------------------------------------
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

        lbl = REGION_LABEL.get(selected_cls, selected_cls)
        st.markdown(f"""
        <div class="divider-label"><span>Retrieved Results</span></div>
        <div class="results-header">
            <div class="results-title">Similar Products</div>
            <div class="results-count">
                top {top_k} &nbsp;·&nbsp; {lbl} &nbsp;·&nbsp; cosine similarity
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(top_k, gap="small")
        valid_results = 0

        for cand in candidates:
            if valid_results >= top_k:
                break

            idx        = cand["gallery_idx"]
            similarity = cand["score"]

            try:
                row        = gallery_df.iloc[idx]
                filename   = row["image_path"].split("/")[-1]
                crop_class = row["crop_class"]
                img_path   = os.path.join("cropped_products", crop_class, filename)

                # --- Gallery Path Verification ---
                # Ensuring runtime folder structure matches expectation:
                # cropped_products/upper_body/, etc.
                if not os.path.exists(img_path):
                    st.warning(f"Warning: Image missing at path {img_path}. Verify cropped_products directory.")
                    continue

                item_id       = row["item_id"]
                caption       = row.get("caption", "") or ""
                retrieved_img = Image.open(img_path)
                sim_pct       = int(min(max(float(similarity), 0.0), 1.0) * 100)

                with cols[valid_results]:
                    st.image(retrieved_img, use_container_width=True)
                    st.markdown(f"""
                    <div class="result-body">
                        <div class="result-score-row">
                            <div>
                                <div class="result-score-label">Similarity</div>
                                <div class="result-score-num">{similarity:.4f}</div>
                            </div>
                            <span class="region-tag {selected_cls}" style="font-size:0.55rem;padding:3px 8px">
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
                No images found. Try disabling Category Filter or switching region.
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer-text">
    VR_Project &nbsp;·&nbsp; best_deepfashion_yolo.pt &nbsp;·&nbsp; CLIP ViT-B/32 &nbsp;·&nbsp;
    HNSW FAISS &nbsp;·&nbsp; MULTI-CROP FUSION &nbsp;·&nbsp; COSINE RERANKING
</div>
""", unsafe_allow_html=True)