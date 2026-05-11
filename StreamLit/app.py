import os
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import torch
import open_clip

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
- ViT-B/32
- 512-D Embeddings
- FAISS Similarity Search
- DeepFashion Dataset
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Evaluation")

st.sidebar.markdown("""
- Recall@5 : 76.7%
- Recall@10 : 80.6%
- Recall@15 : 83.3%
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
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    embeddings = np.load("embeddings.npy")

    metadata = pd.read_csv("metadata_with_embeddings.csv")

    embeddings = embeddings.astype("float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    embeddings = embeddings / norms

    gallery_df = metadata[
        metadata["split"] == "gallery"
    ].reset_index(drop=True)

    gallery_indices = gallery_df["embedding_index"].values

    gallery_embeddings = embeddings[gallery_indices]

    gallery_embeddings = gallery_embeddings.astype("float32")

    return embeddings, gallery_df, gallery_embeddings

embeddings, gallery_df, gallery_embeddings = load_data()

# =========================================================
# BUILD FAISS INDEX
# =========================================================

@st.cache_resource
def build_faiss_index(gallery_embeddings):

    index = faiss.IndexFlatIP(gallery_embeddings.shape[1])

    index.add(gallery_embeddings)

    return index

gallery_index = build_faiss_index(gallery_embeddings)

# =========================================================
# IMAGE ENCODING
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
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

# =========================================================
# RETRIEVAL
# =========================================================

if uploaded_file is not None:

    query_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 3])

    with col1:

        st.image(
            query_image,
            caption="Query Image",
            use_container_width=True
        )

    with col2:

        st.markdown("## Retrieving Similar Products...")

        query_embedding = encode_image(query_image)

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

            original_path = gallery_df.iloc[idx]["image_path"]

            filename = original_path.split("/")[-1]

            crop_class = gallery_df.iloc[idx]["crop_class"]

            img_path = os.path.join(
                "cropped_products",
                crop_class,
                filename
)

            if not os.path.exists(img_path):
                continue

            item_id = gallery_df.iloc[idx]["item_id"]

            caption = gallery_df.iloc[idx]["caption"]

            similarity = scores[0][i]

            retrieved_img = Image.open(img_path)

            with cols[valid_results]:

                st.image(
                    retrieved_img,
                    use_container_width=True
                )

                st.markdown(
                    f"<div class='result-box'>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"### #{valid_results + 1}"
                )

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

                st.markdown(
                    "</div>",
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
    Visual Retrieval using CLIP embeddings and FAISS indexing
    </span>
    </center>
    """,
    unsafe_allow_html=True
)