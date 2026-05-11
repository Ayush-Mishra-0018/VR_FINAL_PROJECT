import os
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import torch
import open_clip

from ultralytics import YOLO
from PIL import Image

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

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

.title-text {
    font-size: 3.2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
}

.subtitle-text {
    font-size: 1.1rem;
    color: #B0B3B8;
    margin-bottom: 2rem;
}

.similarity-text {
    color: #00FFAA;
    font-weight: 700;
    font-size: 0.95rem;
}

.caption-text {
    color: #C9D1D9;
    font-size: 0.9rem;
}

.result-box {
    background-color: #161A22;
    padding: 0.8rem;
    border-radius: 12px;
    margin-top: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Retrieval Settings")

top_k = st.sidebar.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model")

st.sidebar.markdown("""
- YOLOv8 Detection
- CLIP ViT-B/32
- HNSW FAISS Retrieval
- DeepFashion Dataset
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Evaluation Metrics")

st.sidebar.markdown("""
- Recall@5 : 41.0%
- Recall@10 : 46.4%
- Recall@15 : 49.1%

- mAP@5 : 11.6%
- mAP@10 : 12.3%
- mAP@15 : 12.6%

- NDCG@5 : 18.8%
- NDCG@10 : 18.7%
- NDCG@15 : 19.2%
""")

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title-text">🧥 Visual Fashion Retrieval System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">Upload a clothing image to retrieve visually similar fashion products.</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD CLIP MODEL
# =========================================================

@st.cache_resource
def load_clip_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )

    model = model.to(device)

    model.eval()

    return model, preprocess, device

model, preprocess, device = load_clip_model()

# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    model = YOLO("yolov8n.pt")

    return model

yolo_model = load_yolo_model()

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    embeddings = np.load("embeddings.npy")

    metadata = pd.read_csv(
        "metadata_with_embeddings.csv"
    )

    embeddings = embeddings.astype("float32")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / norms

    gallery_df = metadata[
        metadata["split"] == "gallery"
    ].reset_index(drop=True)

    gallery_indices = gallery_df[
        "embedding_index"
    ].values

    gallery_embeddings = embeddings[
        gallery_indices
    ]

    gallery_embeddings = gallery_embeddings.astype(
        "float32"
    )

    return embeddings, gallery_df, gallery_embeddings

embeddings, gallery_df, gallery_embeddings = load_data()

# =========================================================
# BUILD HNSW FAISS INDEX
# =========================================================

@st.cache_resource
def build_faiss_index(gallery_embeddings):

    dim = gallery_embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dim,
        32
    )

    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16

    index.add(gallery_embeddings)

    return index

gallery_index = build_faiss_index(
    gallery_embeddings
)

# =========================================================
# ENCODE IMAGE
# =========================================================

def encode_image(image):

    image_tensor = preprocess(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        features = model.encode_image(image_tensor)

    features = features.cpu().numpy()

    features = features.astype("float32")

    features /= np.linalg.norm(features)

    return features

# =========================================================
# YOLO DETECTION + CROP
# =========================================================

def detect_and_crop(image):

    results = yolo_model(image)

    boxes = results[0].boxes

    if len(boxes) == 0:
        return image

    largest_box = None
    largest_area = 0

    for box in boxes:

        cls_id = int(box.cls[0])

        # person class
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0]

        area = (x2 - x1) * (y2 - y1)

        if area > largest_area:

            largest_area = area

            largest_box = [x1, y1, x2, y2]

    if largest_box is None:
        return image

    x1, y1, x2, y2 = map(int, largest_box)

    cropped = image.crop((x1, y1, x2, y2))

    return cropped

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

# =========================================================
# MAIN PIPELINE
# =========================================================

if uploaded_file is not None:

    query_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("## Uploaded Image")

    st.image(
        query_image,
        width=300
    )

    st.markdown("---")

    st.markdown("## YOLO Detection Output")

    cropped_image = detect_and_crop(
        query_image
    )

    st.image(
        cropped_image,
        width=300
    )

    confirm = st.button("Confirm Crop")

    # =====================================================
    # RETRIEVAL
    # =====================================================

    if confirm:

        with st.spinner(
            "Retrieving similar products..."
        ):

            query_embedding = encode_image(
                cropped_image
            )

            scores, indices = gallery_index.search(
                query_embedding,
                top_k * 5
            )

        st.success("Retrieval Complete")

        st.markdown("---")

        st.markdown("## Retrieved Results")

        cols = st.columns(top_k)

        valid_results = 0

        for i, idx in enumerate(indices[0]):

            try:

                original_path = gallery_df.iloc[idx][
                    "image_path"
                ]

                filename = original_path.split("/")[-1]

                crop_class = gallery_df.iloc[idx][
                    "crop_class"
                ]

                img_path = os.path.join(
                    "cropped_products",
                    crop_class,
                    filename
                )

                if not os.path.exists(img_path):
                    continue

                item_id = gallery_df.iloc[idx][
                    "item_id"
                ]

                caption = gallery_df.iloc[idx][
                    "caption"
                ]

                similarity = scores[0][i]

                retrieved_img = Image.open(
                    img_path
                )

                with cols[valid_results]:

                    st.image(
                        retrieved_img,
                        use_container_width=True
                    )

                    st.markdown(
                        f"### #{valid_results + 1}"
                    )

                    st.progress(float(similarity))

                    st.markdown(
                        f"<div class='similarity-text'>Similarity: {similarity:.4f}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"<div class='caption-text'><b>Item ID:</b> {item_id}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"<div class='caption-text'>{caption}</div>",
                        unsafe_allow_html=True
                    )

                valid_results += 1

                if valid_results >= top_k:
                    break

            except:
                continue

        if valid_results == 0:

            st.warning(
                "No matching images could be displayed."
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <span style='color:gray'>
    Visual Retrieval using YOLO, CLIP embeddings and HNSW-based FAISS indexing
    </span>
    </center>
    """,
    unsafe_allow_html=True
)