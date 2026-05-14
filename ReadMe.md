# 🛍️ Visual Product Search Engine

**Visual Recognition Course Project**

An end-to-end multimodal query-by-image search system that allows users to upload a clothing image and retrieves visually and semantically similar products from a large fashion catalog.

---

## 📖 Problem Statement & Motivation

Traditional e-commerce platforms primarily rely on keyword-based searches, which suffer from vocabulary mismatch and inconsistent metadata. When users discover a clothing item they like through an image, they often struggle to describe it textually to find it online.

This project addresses this by implementing a **query-by-image product search system**. Using advanced deep learning and computer vision techniques, the system automatically detects garments, generates semantic descriptions, and retrieves visually similar products using cross-modal embeddings and approximate nearest neighbor (ANN) search.

---

## 🚀 Key Features

* **Garment Detection & Cropping**: Uses YOLO-based object detection to automatically identify and crop primary clothing regions, eliminating background noise.
* **Semantic Captioning**: Generates rich natural language descriptions using BLIP-2 to capture semantic attributes like color, sleeve style, fit, and texture.
* **Multimodal Embeddings**: Fuses visual and text embeddings using OpenAI's CLIP (with a fine-tuned vision encoder via supervised contrastive loss).
* **Category-Aware Retrieval**: Implements FAISS HNSW index search with precise semantic filtering across coarse garment categories (Upper Body, Lower Body, Full Body).
* **Exact Cosine Re-ranking**: Post-retrieval re-ranking to correct HNSW approximation errors for precise Top-K matches.
* **Interactive UI**: A local Streamlit application providing real-time crop-confirmation and end-to-end visual retrieval.

---

## 🧠 System Architecture

The pipeline is split into an Offline Indexing phase and an Online Query phase.

### Offline Indexing Pipeline
1. **Product Localization (YOLO)**: A YOLOv8 model detects apparel regions and crops the bounding box with the highest confidence.
2. **Semantic Captioning (BLIP-2)**: Generates detailed garment descriptions for each cropped image.
3. **Cross-Modal Embedding (CLIP)**: Cropped images and captions are encoded, fused into a single normalized vector, and indexed.
4. **Vector Indexing (FAISS)**: Stored in an HNSW-based Approximate Nearest Neighbor (ANN) index for fast, scalable retrieval.

### Online Retrieval Pipeline
1. **Query Upload & Preprocessing**: The user uploads an image, and the system runs the Hugging Face `yolos-fashionpedia` detector to extract the garment region.
2. **Crop Confirmation**: The interface pauses, allowing the user to confirm the detected crop to prevent searching on irrelevant data.
3. **Query Encoding & Augmentation**: The confirmed crop is optionally augmented (inward cropping, brightness adjustments), encoded by CLIP, and L2-normalized.
4. **Candidate Retrieval & Re-ranking**: Searches the FAISS index restricted by the detected garment category, followed by exact cosine similarity re-ranking and duplicate removal.

---

## 📁 Repository Structure

```text
VR_FINAL_PROJECT/
│
├── StreamLit/                      # Interactive Demo Application
│   ├── app.py                      # Main Streamlit app script
│   ├── batch_eval.py               # Batch evaluation script for metrics
│   ├── requirements.txt            # Python dependencies
│   ├── DETAILED_STREAMLIT_README.md# Detailed Streamlit documentation
│   └── ... (models, embeddings, and gallery crops)
│
├── YOLO/                           # YOLO Detection Pipeline
│   ├── TrainedYOLOfor3bb           # YOLO fine-tuning notebooks/scripts
│   └── YOLO_train                  # Training code for DeepFashion dataset
│
├── Experiment/                     # Multimodal Experimentation
│   ├── blip/                       # BLIP-2 caption generation scripts
│   └── clip/                       # CLIP fine-tuning and contrastive loss code
│
├── Evaluations/                    # Evaluation & Metrics
│   ├── validations.ipynb           # Testing and metric validation notebook
│   └── eval_results_full.csv       # Precomputed retrieval metrics (mAP, Recall, NDCG)
│
├── Report.pdf                      # Comprehensive Technical Report
└── VR-Final-Project.pdf            # Problem Statement & Rubric Documentation
```

---

## 🛠️ Setup & Installation

### Requirements
* Python 3.10+
* 8GB+ RAM
* GPU recommended but fully functional on CPU.

### Installation
1. Clone the repository and navigate to the Streamlit directory:
   ```bash
   cd StreamLit
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*Note: Before running the application, ensure `embeddings.npy`, `metadata_with_embeddings.csv`, `yolov8n.pt`, and the `cropped_products/` directory are placed inside the `StreamLit/` folder as described in the [Detailed Streamlit README](./StreamLit/DETAILED_STREAMLIT_README.md).*

---

## 🖥️ Running the Application

To launch the interactive Streamlit demo:
```bash
source venv/bin/activate
streamlit run app.py
```
Then, open `http://localhost:8501` in your browser.

**Workflow**:
1. Upload a fashion image.
2. View the YOLO detected region.
3. Click **Confirm Crop**.
4. View the Top-K retrieved visual matches along with similarity scores and generated metadata!

---

## 📊 Evaluation & Performance

The model's retrieval capability was evaluated on the **DeepFashion In-Shop Clothes Retrieval Dataset**. 
Metrics computed include **Recall@K**, **mAP@K**, and **NDCG@K** for $K \in \{5, 10, 15\}$.

You can run the evaluation script locally to reproduce results:
```bash
cd StreamLit
python batch_eval.py
```

---

## 👥 Team Members

* **Ayush Mishra** - IMT2023129
* **Kartikeya Dimri** - IMT2023126
* **Harsh Sinha** - IMT2023571
* **Nipun Verma** - IMT2023591

