import numpy as np
import pandas as pd
import faiss
import torch
import open_clip

from tqdm import tqdm

# =========================================================
# LOAD CLIP MODEL
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)

model.eval()

print("CLIP Model Loaded")

# =========================================================
# LOAD DATA
# =========================================================

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

print("Embeddings Loaded")

# =========================================================
# SPLITS
# =========================================================

gallery_df = metadata[
    metadata["split"] == "gallery"
].reset_index(drop=True)

query_df = metadata[
    metadata["split"] == "query"
].reset_index(drop=True)

# =========================================================
# GALLERY EMBEDDINGS
# =========================================================

gallery_indices = gallery_df[
    "embedding_index"
].values

gallery_embeddings = embeddings[
    gallery_indices
]

gallery_embeddings = gallery_embeddings.astype(
    "float32"
)

# =========================================================
# BUILD HNSW FAISS INDEX
# =========================================================

dim = gallery_embeddings.shape[1]

gallery_index = faiss.IndexHNSWFlat(
    dim,
    32
)

gallery_index.hnsw.efConstruction = 40
gallery_index.hnsw.efSearch = 16

gallery_index.add(gallery_embeddings)

print("HNSW FAISS Index Built")

# =========================================================
# METRIC FUNCTIONS
# =========================================================

def recall_at_k(
    relevant,
    retrieved,
    k
):

    retrieved_k = retrieved[:k]

    return int(
        any(r in relevant for r in retrieved_k)
    )


def average_precision_at_k(
    relevant,
    retrieved,
    k
):

    retrieved_k = retrieved[:k]

    score = 0

    hits = 0

    for i, item in enumerate(retrieved_k):

        if item in relevant:

            hits += 1

            score += hits / (i + 1)

    if len(relevant) == 0:
        return 0

    return score / len(relevant)


def dcg_at_k(
    relevant,
    retrieved,
    k
):

    retrieved_k = retrieved[:k]

    dcg = 0

    for i, item in enumerate(retrieved_k):

        if item in relevant:

            dcg += 1 / np.log2(i + 2)

    return dcg


def ndcg_at_k(
    relevant,
    retrieved,
    k
):

    ideal = dcg_at_k(
        relevant,
        relevant,
        min(k, len(relevant))
    )

    if ideal == 0:
        return 0

    return dcg_at_k(
        relevant,
        retrieved,
        k
    ) / ideal

# =========================================================
# EVALUATION STORAGE
# =========================================================

recalls = {
    5: [],
    10: [],
    15: []
}

aps = {
    5: [],
    10: [],
    15: []
}

ndcgs = {
    5: [],
    10: [],
    15: []
}

print()

print("=" * 60)

print("RUNNING BATCH EVALUATION")

print("=" * 60)

print()

# =========================================================
# QUERY LOOP
# =========================================================

for i in tqdm(range(len(query_df))):

    query_item_id = query_df.iloc[i][
        "item_id"
    ]

    relevant = list(

        gallery_df[
            gallery_df["item_id"]
            == query_item_id
        ]["item_id"]

    )

    query_embedding_idx = query_df.iloc[i][
        "embedding_index"
    ]

    query_vector = embeddings[
        query_embedding_idx
    ].reshape(1, -1)

    scores, indices = gallery_index.search(
        query_vector,
        15
    )

    retrieved = [

        gallery_df.iloc[idx]["item_id"]

        for idx in indices[0]

    ]

    for k in [5, 10, 15]:

        recalls[k].append(

            recall_at_k(
                relevant,
                retrieved,
                k
            )

        )

        aps[k].append(

            average_precision_at_k(
                relevant,
                retrieved,
                k
            )

        )

        ndcgs[k].append(

            ndcg_at_k(
                relevant,
                retrieved,
                k
            )

        )

# =========================================================
# FINAL RESULTS
# =========================================================

print()

print("=" * 60)

print("FINAL METRICS")

print("=" * 60)

print()

for k in [5, 10, 15]:

    print(
        f"Recall@{k}:",
        round(np.mean(recalls[k]), 4)
    )

    print(
        f"mAP@{k}:",
        round(np.mean(aps[k]), 4)
    )

    print(
        f"NDCG@{k}:",
        round(np.mean(ndcgs[k]), 4)
    )

    print()

print("=" * 60)

print("BATCH EVALUATION COMPLETE")

print("=" * 60)