# 📷 Photo Scanner — AI-Powered Local Photo Organizer

A fully **offline**, AI-powered photo management system. Scan your photo library, search by natural language, detect duplicates, and recognize faces — all on your own machine, with no cloud uploads.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Smart Scanning** | Recursively indexes images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`) with incremental re-scan support |
| **AI Visual Search** | Natural language queries using CLIP (`clip-ViT-B-32`) — e.g. *"dog on the beach"*, *"birthday cake"* |
| **OCR Text Search** | Auto-detects documents (receipts, invoices, letters) via a gatekeeper CLIP threshold and extracts text with PaddleOCR |
| **Hybrid Scoring** | Dynamically weights Visual + OCR + Metadata scores based on query intent |
| **Metadata Search** | Searches by camera, date, GPS location, device, shutter speed, ISO, aperture |
| **Duplicate Detection** | Finds exact, near-exact, and semantically similar duplicates using pHash (BK-tree) + CLIP embeddings (FAISS) |
| **Face Recognition** | Detects faces with InsightFace/ArcFace and clusters them into person identities using DBSCAN |
| **Feedback Learning** | Records click/positive/negative feedback to re-rank future search results |
| **FAISS Indexing** | Sub-linear ANN search over the full embedding space for fast retrieval |
| **Privacy First** | 100% local — no internet required after initial model download |
| **Mobile Export** | Exports quantized ONNX model + SQLite DB for mobile app integration |

---

## 🗂 Project Structure

```
Photo_scanner/
├── main.py                  # CLI entry point (scan / search / dedupe / face commands)
├── search_config.py         # Tunable search weights and thresholds
├── export_to_mobile.py      # Exports ONNX model + DB for mobile apps
├── pyproject.toml           # Poetry dependency manifest
└── src/
    ├── scanner.py           # Core scanning pipeline (batch, OCR gating, face detection)
    ├── database.py          # SQLite CRUD — photos, faces, persons tables
    ├── model_handler.py     # CLIP model wrapper (sentence-transformers)
    ├── duplicate_detector.py# pHash (BK-tree) + CLIP embedding (FAISS) dedup engine
    ├── faiss_index.py       # FAISS IVF index build / load / search
    ├── query_analyzer.py    # Intent detection → dynamic weight selection
    ├── metadata_scorer.py   # Metadata field scoring for hybrid search
    ├── ocr_handler.py       # PaddleOCR text extraction
    ├── image_analyzer.py    # Visual attribute analysis (YOLO-based)
    ├── face_detector.py     # InsightFace face detection + ArcFace embedding
    ├── person_clusterer.py  # DBSCAN clustering → person identity groups
    ├── feedback_handler.py  # User feedback storage + penalty/boost computation
    ├── search_cache.py      # On-disk text-embedding cache (avoids repeated model load)
    └── learning_engine.py   # Adaptive weight learning from feedback history
```

---

## ⚙️ Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12 – 3.14 |
| [Poetry](https://python-poetry.org/docs/#installation) | Latest |
| RAM | 8 GB+ recommended |
| Disk | ~3 GB for models + your photo DB |

> **Windows users**: The `insightface` package requires the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Install them before running `poetry install`.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/arjav0112/Photo_scanner_model.git
cd Photo_scanner_model

# 2. Install all dependencies (creates an isolated virtualenv automatically)
poetry install
```

On the first `scan` run, the CLIP model (`clip-ViT-B-32`) is downloaded from HuggingFace and cached locally.

---

## 📖 Usage

All commands are run via `poetry run python main.py <command>`.

### 1. Scan a Directory

Indexes all photos, extracts embeddings, metadata, OCR text, and detects faces.

```bash
poetry run python main.py scan "C:\Users\YourName\Pictures"

# Windows — scan current folder
poetry run python main.py scan .
```

- **Incremental**: Only new/modified files are re-processed on subsequent runs.
- A `photos.db` SQLite database and FAISS index are created/updated automatically.

---

### 2. Search Photos

```bash
poetry run python main.py search "sunset at the beach"
poetry run python main.py search "invoice January 2024"
poetry run python main.py search "photos taken in Delhi"
```

The engine auto-detects query intent (visual / OCR / metadata / location) and adjusts weights accordingly. Results are printed to the terminal and saved to `search_results.log`.

**Interactive feedback** — after results appear:
- Type `1`–`9` to open an image
- `y` / `n` to mark it helpful or not (improves future results)
- `q` to quit

---

### 3. Find & Remove Duplicate Images

```bash
# Interactive review (default)
poetry run python main.py dedupe

# Auto-mark all duplicates in the database (no deletion)
poetry run python main.py dedupe --auto-mark

# Delete duplicate files from disk (IRREVERSIBLE)
poetry run python main.py dedupe --delete

# List all previously flagged duplicates
poetry run python main.py dedupe --list

# Tune thresholds
poetry run python main.py dedupe --phash-threshold 5 --embedding-threshold 0.92
```

**How it works:**
- **Phase 1 — pHash (BK-tree)**: Detects pixel-identical and near-identical images in O(n log n) using perceptual hashing + Hamming distance.
- **Phase 2 — Embeddings (FAISS)**: Detects visually similar images (same scene, different quality/crop) using CLIP cosine similarity.
- Groups are labeled `EXACT` / `NEAR_EXACT` / `SEMANTIC`.

| Interactive action | Effect |
|---|---|
| `m` | Mark as duplicate in DB |
| `d` | Delete from disk + DB |
| `s` | Skip this group |
| `q` | Quit |

---

### 4. Face Recognition & Grouping

Cluster all detected faces into person identities:

```bash
poetry run python main.py group-faces

# Reset and recluster from scratch
poetry run python main.py group-faces --reset
```

Assign a name to a person:
```bash
poetry run python main.py name-person 1 "Alice"
```

Find all photos containing a person:
```bash
poetry run python main.py search-person "Alice"
poetry run python main.py search-person 1
```

---

## 🔧 Configuration

Edit `search_config.py` to tune scoring behaviour:

```python
# Relative threshold for result filtering (0.0 = show all, 0.5 = top half)
"relative_threshold": 0.30

# Maximum results returned
"max_results": 10

# Minimum cosine score to include a result
"floor_ratio": 0.70
```

Dedupe thresholds (can also be set via CLI flags):

| Setting | Default | Meaning |
|---|---|---|
| `--phash-threshold` | 10 | Max Hamming bits to be "near-exact" (0 = pixel-identical) |
| `--embedding-threshold` | 0.90 | Min cosine similarity to be "semantic duplicate" |

---

## 📦 Dependencies (key packages)

| Package | Purpose |
|---|---|
| `sentence-transformers` | CLIP visual + text embeddings |
| `faiss-cpu` | ANN index for fast similarity search |
| `paddleocr` | Text extraction from documents |
| `insightface` | Face detection + ArcFace embeddings |
| `ultralytics` | YOLOv8 object detection (visual analysis) |
| `tensorflow` | Model backend |
| `pillow`, `numpy` | Image processing |
| `reverse-geocoder` | Offline GPS → location name |
| `scipy` | DBSCAN clustering for face grouping |

---

## 🗄 Database Schema

`photos.db` (SQLite) — three tables:

- **`photos`** — path, filename, size, mtime, CLIP embedding, metadata JSON, OCR text, pHash, duplicate flags
- **`faces`** — bounding box, ArcFace embedding, age estimate, gender, person_id → photos FK
- **`persons`** — cluster ID, optional human name, representative face
