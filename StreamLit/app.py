import os
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import torch
import open_clip

from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageEnhance

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Visual Fashion Retrieval",
    page_icon="🧥",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .main {
    background-color: #080B12;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.title-text {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #F0F0F0;
    letter-spacing: -1px;
    line-height: 1.1;
}

.title-accent {
    color: #7DF9C4;
}

.subtitle-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #556;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
    text-transform: uppercase;
}

.tag-pill {
    display: inline-block;
    background: #11151F;
    border: 1px solid #1E2535;
    color: #7DF9C4;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.04em;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3E4A60;
    margin-bottom: 0.5rem;
}

.similarity-bar-wrap {
    background: #11151F;
    border-radius: 4px;
    height: 4px;
    margin: 6px 0;
    overflow: hidden;
}

.similarity-bar {
    background: linear-gradient(90deg, #7DF9C4, #3DD6F5);
    height: 4px;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.result-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4A5568;
    line-height: 1.6;
}

.result-sim {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #7DF9C4;
    font-weight: 500;
}

.result-rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #F0F0F0;
}

.result-caption {
    color: #8A9BB5;
    font-size: 0.82rem;
    line-height: 1.5;
    margin-top: 4px;
}

.crop-zone-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3DD6F5;
    margin-bottom: 4px;
}

.mode-badge {
    display: inline-block;
    background: #0F1928;
    border: 1px solid #1A2D45;
    color: #3DD6F5;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 0.08em;
}

.warning-box {
    background: #1A1400;
    border: 1px solid #3D2E00;
    border-left: 3px solid #F5A623;
    color: #C8941A;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 8px 0;
}

.info-box {
    background: #0A1628;
    border: 1px solid #132140;
    border-left: 3px solid #3DD6F5;
    color: #7FAAC5;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙️ Retrieval Settings")

top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=5)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Query Mode")
query_mode = st.sidebar.radio(
    "Crop Strategy",
    options=["Auto-Detect Region", "Full Body", "Upper Body", "Lower Body", "Manual Upload"],
    index=0,
    help=(
        "Auto-Detect: Uses YOLO keypoints to heuristically split upper/lower body. "
        "Full Body: Encodes entire person crop. "
        "Upper/Lower Body: Manually forces a specific half-crop. "
        "Manual Upload: No YOLO — encode as-is."
    )
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔁 Multi-Crop Fusion")
use_multicrop = st.sidebar.checkbox(
    "Enable Multi-Crop Embedding Fusion",
    value=True,
    help=(
        "Encodes 2–3 crops (original, augmented) and averages embeddings. "
        "Reduces sensitivity to pose/background noise."
    )
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Retrieval Filtering")

filter_by_category = st.sidebar.checkbox(
    "Filter by Detected Category",
    value=True,
    help=(
        "Restricts FAISS search to the gallery subset matching the predicted "
        "clothing category (upper/lower/dress/etc). Prevents cross-category noise."
    )
)

use_reranking = st.sidebar.checkbox(
    "Enable Cosine Reranking",
    value=True,
    help=(
        "After ANN retrieval, reranks results by exact cosine similarity "
        "against the fused query embedding. Corrects HNSW approximation errors."
    )
)

dedup_results = st.sidebar.checkbox(
    "Deduplicate by Item ID",
    value=True,
    help="Removes duplicate item_id entries from results, improving diversity."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model")
st.sidebar.markdown("""
<div class='result-meta'>
• YOLOv8n Detection<br>
• OpenCLIP ViT-B/32<br>
• HNSW FAISS Retrieval<br>
• DeepFashion Dataset
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Evaluation Metrics")
st.sidebar.markdown("""
<div class='result-meta'>
Recall@5 &nbsp;&nbsp;: 41.0%<br>
Recall@10 : 46.4%<br>
Recall@15 : 49.1%<br><br>
mAP@5 &nbsp;&nbsp;&nbsp;&nbsp;: 11.6%<br>
mAP@10 &nbsp;&nbsp;&nbsp;: 12.3%<br>
mAP@15 &nbsp;&nbsp;&nbsp;: 12.6%<br><br>
NDCG@5 &nbsp;&nbsp;&nbsp;: 18.8%<br>
NDCG@10 &nbsp;&nbsp;: 18.7%<br>
NDCG@15 &nbsp;&nbsp;: 19.2%
</div>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title-text">🧥 Visual Fashion <span class="title-accent">Retrieval</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle-text">Upload a clothing image · detect region · retrieve similar products</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD CLIP MODEL
# =========================================================

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    return model, preprocess, device

model, preprocess, device = load_clip_model()

# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy").astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    metadata = pd.read_csv("metadata_with_embeddings.csv")
    gallery_df = metadata[metadata["split"] == "gallery"].reset_index(drop=True)
    gallery_indices = gallery_df["embedding_index"].values
    gallery_embeddings = embeddings[gallery_indices].astype("float32")

    return embeddings, gallery_df, gallery_embeddings

embeddings, gallery_df, gallery_embeddings = load_data()

# =========================================================
# BUILD HNSW FAISS INDEX — TUNED efSearch
# =========================================================

@st.cache_resource
def build_faiss_index(gallery_embeddings):
    dim = gallery_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200   # higher = better graph quality
    index.hnsw.efSearch = 64          # higher = better recall at query time
    index.add(gallery_embeddings)
    return index

gallery_index = build_faiss_index(gallery_embeddings)

# =========================================================
# CATEGORY-AWARE SUBSET INDICES
# =========================================================

@st.cache_data
def build_category_subsets(gallery_df):
    """
    Build per-category FAISS indices (or index mappings) so we can filter
    to upper/lower/full body subsets at retrieval time.
    WHY: Prevents cross-category noise — e.g. lower-body queries retrieving
    shirts — which is the primary driver of semantic drift.
    """
    subsets = {}
    if "crop_class" in gallery_df.columns:
        for cat in gallery_df["crop_class"].unique():
            mask = gallery_df["crop_class"] == cat
            subsets[cat] = gallery_df[mask].index.tolist()
    return subsets

category_subsets = build_category_subsets(gallery_df)

# =========================================================
# ENCODE IMAGE — WITH OPTIONAL AUGMENTATION FUSION
# =========================================================

def encode_image(image: Image.Image, use_fusion: bool = True) -> np.ndarray:
    """
    Encode image to normalized CLIP embedding.

    If use_fusion=True, generates 3 crops:
      1. Original
      2. Slight center crop (removes border/background)
      3. Slightly brightened (reduces lighting bias)
    Averages embeddings → more robust query representation.

    WHY: CLIP is sensitive to background, lighting, and exact crop boundary.
    Averaging a few mild augmentations suppresses these nuisance factors
    without distorting semantics.
    """
    def _encode_single(img):
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor).cpu().numpy().astype("float32")
        feat /= np.linalg.norm(feat) + 1e-8
        return feat

    if not use_fusion:
        return _encode_single(image)

    crops = [image]

    # Crop 2: center 90% crop → removes edge background
    w, h = image.size
    margin_w, margin_h = int(w * 0.05), int(h * 0.05)
    center_crop = image.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    crops.append(center_crop)

    # Crop 3: slight brightness boost (neutralizes dim/overexposed images)
    bright = ImageEnhance.Brightness(image).enhance(1.15)
    crops.append(bright)

    embeddings_list = [_encode_single(c) for c in crops]
    fused = np.mean(embeddings_list, axis=0)
    fused /= np.linalg.norm(fused) + 1e-8
    return fused.astype("float32")

# =========================================================
# CLASSIFY QUERY REGION — UPPER / LOWER / FULL
# =========================================================

def classify_crop_region(box_xyxy, image_height: int) -> str:
    """
    Given a detected person bounding box and image height,
    heuristically decide whether the query focuses on
    upper, lower, or full body.

    WHY: The original code crops just the person but doesn't
    distinguish which garment region is salient. This causes
    CLIP to encode a mix of upper/lower features, leading to
    semantically ambiguous embeddings that retrieve mismatched items.
    """
    x1, y1, x2, y2 = box_xyxy
    box_h = y2 - y1
    top_frac = y1 / image_height
    bot_frac = y2 / image_height

    if top_frac < 0.1 and bot_frac > 0.75:
        return "full"
    if bot_frac < 0.55:
        return "upper"
    if top_frac > 0.4:
        return "lower"
    return "full"

# =========================================================
# YOLO DETECTION + SMART CROP
# =========================================================

def detect_and_crop(image: Image.Image, mode: str = "Auto-Detect Region"):
    """
    Detect person and crop according to selected query mode.

    Modes:
    - Auto-Detect: YOLO person box + heuristic region split
    - Upper Body:  Top 55% of the person crop
    - Lower Body:  Bottom 55% of the person crop
    - Full Body:   Entire person crop
    - Manual:      Return image as-is

    WHY: Forcing a specific crop region resolves the semantic drift
    caused by encoding the entire person when the user intends to
    query upper or lower garments specifically.
    """
    if mode == "Manual Upload":
        return image, "full", None

    results = yolo_model(image)
    boxes = results[0].boxes

    largest_box = None
    largest_area = 0

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
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

    if mode == "Auto-Detect Region":
        region = auto_region
    elif mode == "Upper Body":
        region = "upper"
    elif mode == "Lower Body":
        region = "lower"
    else:  # Full Body
        region = "full"

    if region == "upper":
        crop = person_crop.crop((0, 0, pw, int(ph * 0.55)))
    elif region == "lower":
        crop = person_crop.crop((0, int(ph * 0.45), pw, ph))
    else:
        crop = person_crop

    return crop, region, auto_region

# =========================================================
# MAP DETECTED REGION → GALLERY CATEGORY FILTER
# =========================================================

def get_category_filter(region: str, category_subsets: dict) -> list | None:
    """
    Map the detected body region to a list of row indices in gallery_df
    that should be searched.

    WHY: By restricting FAISS search to only relevant product categories,
    we eliminate cross-category false positives entirely. A lower-body
    query will never retrieve shirts.
    """
    upper_cats = {"upper", "top", "shirt", "jacket", "blouse", "coat", "sweater", "hoodie"}
    lower_cats = {"lower", "bottom", "pants", "jeans", "skirt", "shorts", "trousers"}
    full_cats  = {"full", "dress", "jumpsuit", "romper", "suit", "outerwear"}

    if region == "upper":
        target = upper_cats
    elif region == "lower":
        target = lower_cats
    else:
        return None  # no filter for full — search all

    matched = []
    for cat, idxs in category_subsets.items():
        if any(k in cat.lower() for k in target):
            matched.extend(idxs)

    return matched if matched else None

# =========================================================
# RETRIEVAL WITH RERANKING + DEDUPLICATION
# =========================================================

def retrieve(
    query_embedding: np.ndarray,
    top_k: int,
    category_filter: list | None,
    use_reranking: bool,
    dedup: bool
) -> list[dict]:
    """
    Multi-stage retrieval:
      1. (Optional) Build a filtered sub-index from category rows
      2. FAISS HNSW ANN search with oversampling (top_k * 8)
      3. (Optional) Exact cosine reranking of candidates
      4. (Optional) Deduplicate by item_id

    WHY:
    - Oversampling (top_k * 8) + reranking corrects HNSW approximation
      errors that systematically bias toward structurally similar vectors
      even when semantics differ.
    - Deduplication ensures diverse results when the same product
      appears under multiple gallery entries.
    """
    oversample = top_k * 8

    if category_filter is not None and len(category_filter) > 0:
        # Build a temporary flat index for the filtered subset
        subset_embeddings = gallery_embeddings[category_filter].astype("float32")
        dim = subset_embeddings.shape[1]
        sub_index = faiss.IndexFlatIP(dim)
        sub_index.add(subset_embeddings)
        scores, local_indices = sub_index.search(query_embedding, min(oversample, len(category_filter)))
        # Map local indices back to gallery_df row indices
        global_indices = [category_filter[i] for i in local_indices[0] if i < len(category_filter)]
        raw_scores = scores[0][:len(global_indices)]
    else:
        scores, indices = gallery_index.search(query_embedding, oversample)
        global_indices = indices[0].tolist()
        raw_scores = scores[0]

    candidates = []
    for i, idx in enumerate(global_indices):
        if idx < 0 or idx >= len(gallery_df):
            continue
        candidates.append({
            "gallery_idx": idx,
            "score": float(raw_scores[i])
        })

    if use_reranking and len(candidates) > 0:
        # Exact cosine reranking using precomputed gallery embeddings
        cand_idxs = [c["gallery_idx"] for c in candidates]
        cand_embs = gallery_embeddings[cand_idxs]  # (N, D)
        exact_scores = (cand_embs @ query_embedding.T).squeeze()  # cosine similarity
        for i, c in enumerate(candidates):
            c["score"] = float(exact_scores[i]) if hasattr(exact_scores, "__len__") else float(exact_scores)
        candidates.sort(key=lambda x: x["score"], reverse=True)

    if dedup:
        seen_items = set()
        deduped = []
        for c in candidates:
            iid = gallery_df.iloc[c["gallery_idx"]].get("item_id", c["gallery_idx"])
            if iid not in seen_items:
                seen_items.add(iid)
                deduped.append(c)
        candidates = deduped

    return candidates[:top_k * 3]  # return extra for downstream filtering

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png", "webp"]
)

# =========================================================
# MAIN PIPELINE
# =========================================================

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown('<div class="section-label">Query Image</div>', unsafe_allow_html=True)
        st.image(query_image, width=280)

    with col_info:
        st.markdown('<div class="section-label">Pipeline Settings</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
        Mode &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>{query_mode}</b><br>
        Multi-Crop &nbsp;&nbsp;: {'ON' if use_multicrop else 'OFF'}<br>
        Cat. Filter &nbsp;&nbsp;: {'ON' if filter_by_category else 'OFF'}<br>
        Reranking &nbsp;&nbsp;&nbsp;: {'ON' if use_reranking else 'OFF'}<br>
        Dedup &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {'ON' if dedup_results else 'OFF'}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # DETECTION + CROP
    # =====================================================

    st.markdown('<div class="section-label">Detected Region</div>', unsafe_allow_html=True)

    cropped_image, detected_region, auto_region = detect_and_crop(query_image, mode=query_mode)

    col_crop1, col_crop2 = st.columns([1, 3])
    with col_crop1:
        st.image(cropped_image, caption=f"Crop: {detected_region.upper()}", width=220)

    with col_crop2:
        if auto_region:
            st.markdown(f"""
            <div class='info-box'>
            Auto-detected region: <b>{auto_region.upper()}</b><br>
            Applied crop mode: <b>{detected_region.upper()}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warning-box'>
            No person detected by YOLO. Using original image as query.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='result-meta' style='margin-top:10px'>
        • Change <b>Query Mode</b> in the sidebar to manually override region.<br>
        • <b>Upper Body</b> mode forces top 55% crop.<br>
        • <b>Lower Body</b> mode forces bottom 55% crop.
        </div>
        """, unsafe_allow_html=True)

    confirm = st.button("🔍 Retrieve Similar Products", use_container_width=False)

    # =====================================================
    # RETRIEVAL
    # =====================================================

    if confirm:

        with st.spinner("Encoding query and retrieving..."):

            # Multi-crop embedding fusion
            query_embedding = encode_image(cropped_image, use_fusion=use_multicrop)

            # Category-aware filtering
            cat_filter = None
            if filter_by_category:
                cat_filter = get_category_filter(detected_region, category_subsets)

            # Retrieve with reranking + dedup
            candidates = retrieve(
                query_embedding.reshape(1, -1),
                top_k=top_k,
                category_filter=cat_filter,
                use_reranking=use_reranking,
                dedup=dedup_results
            )

        st.success(f"✓ Retrieved {len(candidates)} candidates · showing top {top_k}")

        if cat_filter is not None:
            st.markdown(f"""
            <div class='info-box'>
            Category filter active: searching <b>{len(cat_filter)}</b> gallery items
            matching region <b>{detected_region.upper()}</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="title-text" style="font-size:1.6rem">Retrieved Results</div>', unsafe_allow_html=True)
        st.markdown("")

        cols = st.columns(top_k)
        valid_results = 0

        for cand in candidates:
            if valid_results >= top_k:
                break

            idx = cand["gallery_idx"]
            similarity = cand["score"]

            try:
                row = gallery_df.iloc[idx]
                original_path = row["image_path"]
                filename = original_path.split("/")[-1]
                crop_class = row["crop_class"]

                img_path = os.path.join("cropped_products", crop_class, filename)
                if not os.path.exists(img_path):
                    continue

                item_id = row["item_id"]
                caption = row.get("caption", "")

                retrieved_img = Image.open(img_path)

                with cols[valid_results]:
                    st.image(retrieved_img, use_container_width=True)

                    sim_pct = min(max(float(similarity), 0.0), 1.0)
                    sim_bar_width = int(sim_pct * 100)

                    st.markdown(f"""
                    <div style='margin-top:6px'>
                        <span class='result-rank'>#{valid_results + 1}</span>
                        <span class='result-sim' style='float:right'>{similarity:.4f}</span>
                    </div>
                    <div class='similarity-bar-wrap'>
                        <div class='similarity-bar' style='width:{sim_bar_width}%'></div>
                    </div>
                    <div class='result-meta'>
                        item: {item_id}<br>
                        class: {crop_class}
                    </div>
                    <div class='result-caption'>{caption[:120] if caption else "—"}</div>
                    """, unsafe_allow_html=True)

                valid_results += 1

            except Exception:
                continue

        if valid_results == 0:
            st.markdown("""
            <div class='warning-box'>
            No matching images could be displayed. Try disabling Category Filter
            or switching Query Mode to Full Body.
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("""
<center>
<span style='font-family: DM Mono, monospace; font-size: 0.7rem; color: #2A3040; letter-spacing: 0.1em;'>
YOLO · CLIP ViT-B/32 · HNSW FAISS · MULTI-CROP FUSION · COSINE RERANKING
</span>
</center>
""", unsafe_allow_html=True)