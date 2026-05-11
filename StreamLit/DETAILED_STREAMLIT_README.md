# Streamlit Demo Application

This directory contains the interactive Streamlit-based demo application for the Visual Fashion Retrieval System.

The application allows users to:
- upload a fashion/product image,
- automatically detect and crop the primary clothing region using YOLOv8,
- confirm the cropped region before retrieval,
- retrieve visually similar fashion products,
- display Top-K retrieval results with similarity scores and metadata.

The demo is designed to showcase the complete end-to-end retrieval pipeline locally.

---

# Features

## Image Upload
Users can upload:
- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

fashion/product images directly through the web interface.

---

## YOLO-based Product Localization
The uploaded image is processed using YOLOv8.

The system:
- detects the primary product/person region,
- crops the detected region,
- displays the cropped output before retrieval.

This helps reduce background clutter and improves retrieval quality.

---

## Crop Confirmation
Before retrieval starts, the cropped image is shown to the user.

The user must click:

```text
Confirm Crop
```

before similarity retrieval begins.

---

## CLIP-based Retrieval
After crop confirmation:
- the cropped image is encoded using CLIP,
- FAISS similarity search is performed,
- visually similar products are retrieved from the gallery database.

---

## Retrieval Results
The application displays:
- retrieved product images,
- similarity scores,
- item IDs,
- product captions/metadata.

---

# Folder Structure

Expected directory structure:

```bash
StreamLit/
│
├── app.py
├── batch_eval.py
├── evaluation_metrics.ipynb
├── embeddings.npy
├── metadata_with_embeddings.csv
├── yolov8n.pt
├── cropped_products/
│   ├── full_body/
│   ├── lower_body/
│   └── upper_body/
│
├── requirements.txt
└── venv/
```

---

# Important Files

## `app.py`
Main Streamlit application.

Handles:
- image upload,
- YOLO detection,
- crop preview,
- CLIP encoding,
- FAISS retrieval,
- result visualization.

---

## `batch_eval.py`
Runs batch evaluation over the dataset and computes:
- Recall@K
- mAP@K
- NDCG@K

for:
- K = 5
- K = 10
- K = 15

---

## `evaluation_metrics.ipynb`
Notebook version of metric computation and retrieval evaluation.

Useful for:
- experimentation,
- debugging,
- metric verification.

---

## `embeddings.npy`
Contains precomputed CLIP embeddings for the dataset.

Required for retrieval.

Without this file:
- FAISS retrieval cannot run.

---

## `metadata_with_embeddings.csv`
Contains:
- item IDs,
- captions,
- dataset splits,
- embedding indices,
- image metadata.

Required for mapping retrieval results to actual images.

---

## `cropped_products/`
Contains gallery images used during retrieval visualization.

This folder is REQUIRED.

Without this folder:
- retrieval images cannot be displayed in Streamlit.

The application expects this structure:

```bash
cropped_products/
├── full_body/
├── lower_body/
└── upper_body/
```

Each folder should contain corresponding cropped product images.

---

## `yolov8n.pt`
YOLOv8 model weights used for product/person detection.

Required for automatic cropping.

Without this file:
- YOLO preprocessing will fail.

---

# System Requirements

Recommended:
- Python 3.10+
- Ubuntu/Linux
- 8GB+ RAM

GPU is optional.

The project also works on CPU.

---

# Setup Instructions

## Step 1 — Clone Repository

```bash
git clone <repository-link>
cd StreamLit
```

---

## Step 2 — Create Virtual Environment

```bash
python3 -m venv venv
```

---

## Step 3 — Activate Environment

### Linux / Ubuntu

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Files Before Running

The following files/directories MUST exist inside the `StreamLit/` directory:

```bash
embeddings.npy
metadata_with_embeddings.csv
cropped_products/
yolov8n.pt
```

---

# Downloading Required Files

The repository alone is NOT sufficient to run the demo application.

Users must additionally obtain:

- `embeddings.npy`
- `metadata_with_embeddings.csv`
- `cropped_products/`
- `yolov8n.pt`

before running the application.

---

# cropped_products Folder

The `cropped_products/` folder contains gallery images used during retrieval visualization.

This folder is large and may not be included directly in the repository.

Expected structure:

```bash
cropped_products/
├── full_body/
├── lower_body/
└── upper_body/
```

Place the folder directly beside:

```bash
app.py
```

inside the `StreamLit/` directory.

---

# Running the Streamlit Application

Inside the `StreamLit/` directory:

```bash
source venv/bin/activate
streamlit run app.py
```

---

# Accessing the Application

After launching Streamlit, open:

```text
http://localhost:8501
```

in a browser.

---

# Streamlit Workflow

The application workflow is:

```text
Upload Image
↓
YOLO Detection
↓
Display Cropped Region
↓
Confirm Crop
↓
CLIP Encoding
↓
FAISS Similarity Search
↓
Top-K Retrieval Results
```

---

# Running Batch Evaluation

To evaluate retrieval performance over the dataset:

```bash
source venv/bin/activate
python batch_eval.py
```

This computes:
- Recall@5/10/15
- mAP@5/10/15
- NDCG@5/10/15

---

# Example Evaluation Output

```text
Recall@5: 0.4271
mAP@5: 0.1199
NDCG@5: 0.1948

Recall@10: 0.4880
mAP@10: 0.1275
NDCG@10: 0.1950

Recall@15: 0.5188
mAP@15: 0.1309
NDCG@15: 0.2004
```

---

# Troubleshooting

## Streamlit does not start

Ensure virtual environment is activated:

```bash
source venv/bin/activate
```

---

## Missing gallery images

Verify that:

```bash
cropped_products/
```

exists and contains:
- `upper_body/`
- `lower_body/`
- `full_body/`

---

## Missing embeddings error

Ensure:

```bash
embeddings.npy
```

exists in the project directory.

---

## YOLO model not found

Ensure:

```bash
yolov8n.pt
```

exists beside:

```bash
app.py
```

---

# Tech Stack

- Streamlit
- YOLOv8
- OpenCLIP
- FAISS
- NumPy
- Pandas
- PyTorch
