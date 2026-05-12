import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# NOTE: CLIP model intentionally NOT loaded here.
# batch_eval uses pre-computed embeddings from embeddings.npy directly.
# Loading the model wasted ~1-2 GB of memory and added startup time for no reason.

import faiss
import numpy as np
import pandas as pd

from tqdm import tqdm

# =========================================================
# LOAD DATA
# =========================================================

embeddings = np.load("embeddings.npy").astype("float32")

# Normalize in one shot
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

metadata = pd.read_csv("metadata_with_embeddings.csv")

gallery_df = metadata[metadata["split"] == "gallery"].reset_index(drop=True)
query_df   = metadata[metadata["split"] == "query"].reset_index(drop=True)

gallery_embeddings = embeddings[gallery_df["embedding_index"].values].astype("float32")

print("=" * 60)
print("DATA LOADED")
print("=" * 60)
print(f"  Embeddings : {embeddings.shape}")
print(f"  Queries    : {len(query_df)}")
print(f"  Gallery    : {len(gallery_df)}")
print()

# =========================================================
# PRECOMPUTE RELEVANT SET PER ITEM ID
# FIX: was re-scanning gallery_df inside the loop (O(N) per query).
# Now a single pass builds a dict: item_id -> set of gallery row indices.
# Membership check is then O(1) per candidate instead of O(N).
# =========================================================

# We need gallery *row indices* (position in gallery_df) as ground truth,
# because retrieved[] is built from gallery row indices returned by FAISS.
item_to_gallery_rows: dict[str, set] = {}
for row_idx, item_id in enumerate(gallery_df["item_id"]):
    item_to_gallery_rows.setdefault(item_id, set()).add(row_idx)

# =========================================================
# BUILD HNSW FAISS INDEX
# FIX: efSearch raised from 16 → 128.
# efSearch=16 is extremely greedy — the graph search skips many good neighbors,
# directly suppressing Recall, mAP, and NDCG numbers at eval time.
# efSearch=128 gives near-exact results on a 15k gallery with negligible extra cost.
# =========================================================

dim = gallery_embeddings.shape[1]
gallery_index = faiss.IndexHNSWFlat(dim, 32)
gallery_index.hnsw.efConstruction = 200   # was 40 — higher = better index quality
gallery_index.hnsw.efSearch = 128         # was 16 — higher = more accurate search
gallery_index.add(gallery_embeddings)

print("=" * 60)
print("HNSW FAISS INDEX BUILT")
print("=" * 60)
print(f"  Vectors    : {gallery_index.ntotal}")
print(f"  efSearch   : {gallery_index.hnsw.efSearch}")
print()

# =========================================================
# METRIC FUNCTIONS
# All functions receive:
#   relevant_set : set of gallery row indices that are ground-truth matches
#   retrieved    : list of gallery row indices returned by FAISS, ranked best-first
#   k            : cutoff
# =========================================================

def recall_at_k(relevant_set: set, retrieved: list, k: int) -> int:
    """
    Binary: 1 if at least one relevant item appears in top-k, else 0.
    Unchanged from original — logic was correct.
    """
    return int(any(r in relevant_set for r in retrieved[:k]))


def average_precision_at_k(relevant_set: set, retrieved: list, k: int) -> float:
    """
    AP@K = (1 / min(|R|, K)) * sum_{i=1}^{K} P@i * rel(i)

    FIX: original code divided by len(relevant) which can exceed k.
    When |R| > k the denominator was too large, making AP@K artificially small
    (e.g. half the true value when |R| = 2k).
    Standard IR definition caps the denominator at min(|R|, k).
    """
    n_relevant = len(relevant_set)
    if n_relevant == 0:
        return 0.0

    score = 0.0
    hits  = 0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant_set:
            hits  += 1
            score += hits / (i + 1)

    return score / min(n_relevant, k)   # <- was: score / len(relevant)


def dcg_at_k(relevant_set: set, retrieved: list, k: int) -> float:
    """
    DCG@K = sum_{i=1}^{K} rel_i / log2(i+1)
    Binary relevance: rel_i in {0, 1}.
    Logic unchanged — was correct.
    """
    return sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(retrieved[:k])
        if item in relevant_set
    )


def ndcg_at_k(relevant_set: set, retrieved: list, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    IDCG@K = DCG of the ideal ranking: all relevant items ranked first.

    FIX (subtle): the ideal list is constructed as a list of min(|R|, k) placeholder
    items that are all "relevant", rather than passing `relevant` as both args to dcg_at_k.
    Original worked by accident only because dcg_at_k checked membership in relevant_set
    while iterating over a list of those same elements — which happened to be correct.
    Made explicit here for clarity and correctness with set-based relevant.
    """
    n_ideal = min(len(relevant_set), k)
    if n_ideal == 0:
        return 0.0

    # Ideal: first n_ideal positions are all hits
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    actual_dcg = dcg_at_k(relevant_set, retrieved, k)

    return actual_dcg / ideal_dcg

# =========================================================
# EVALUATION STORAGE
# =========================================================

K_VALUES = [5, 10, 15]

recalls = {k: [] for k in K_VALUES}
aps     = {k: [] for k in K_VALUES}
ndcgs   = {k: [] for k in K_VALUES}

print("=" * 60)
print("RUNNING BATCH EVALUATION")
print("=" * 60)
print()

# =========================================================
# QUERY LOOP
# =========================================================

MAX_K = max(K_VALUES)

for i in tqdm(range(len(query_df))):

    query_row     = query_df.iloc[i]
    query_item_id = query_row["item_id"]

    # O(1) lookup — was O(N) linear scan inside the loop
    relevant_set  = item_to_gallery_rows.get(query_item_id, set())

    query_vector  = embeddings[query_row["embedding_index"]].reshape(1, -1)

    _, indices    = gallery_index.search(query_vector, MAX_K)

    # retrieved is a list of gallery *row indices* (positions in gallery_df)
    # so ground-truth item_to_gallery_rows values match directly — no extra lookup
    retrieved = [int(idx) for idx in indices[0] if idx >= 0]

    for k in K_VALUES:
        recalls[k].append(recall_at_k(relevant_set, retrieved, k))
        aps[k].append(average_precision_at_k(relevant_set, retrieved, k))
        ndcgs[k].append(ndcg_at_k(relevant_set, retrieved, k))

# =========================================================
# FINAL RESULTS
# =========================================================

print()
print("=" * 60)
print("FINAL METRICS")
print("=" * 60)
print()

for k in K_VALUES:
    print(f"  Recall@{k:<2} : {np.mean(recalls[k]):.4f}")
    print(f"  mAP@{k:<2}    : {np.mean(aps[k]):.4f}")
    print(f"  NDCG@{k:<2}   : {np.mean(ndcgs[k]):.4f}")
    print()

print("=" * 60)
print("BATCH EVALUATION COMPLETE")
print("=" * 60)