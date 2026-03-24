# Photo Scanner — Developer Documentation

> **Written:** March 2026  
> **Author:** Arjav Jain (`jarjav2005@gmail.com`)  
> **Python:** ≥ 3.12, < 3.15 | **Package manager:** Poetry

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Development Journey](#2-development-journey)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Repository Layout](#4-repository-layout)
5. [Dependency Stack](#5-dependency-stack)
6. [Setup & First Run](#6-setup--first-run)
7. [CLI Reference](#7-cli-reference)
8. [Module-by-Module Reference](#8-module-by-module-reference)
   - [main.py](#mainpy)
   - [src/scanner.py](#srcscannerpy)
   - [src/database.py](#srcdatabasepy)
   - [src/model_handler.py](#srcmodel_handlerpy)
   - [src/faiss_index.py](#srcfaiss_indexpy)
   - [src/image_analyzer.py](#srcimage_analyzerpy)
   - [src/ocr_handler.py](#srcocr_handlerpy)
   - [src/face_detector.py](#srcface_detectorpy)
   - [src/person_clusterer.py](#srcperson_clustererpy)
   - [src/duplicate_detector.py](#srcduplicate_detectorpy)
   - [src/query_analyzer.py](#srcquery_analyzerpy)
   - [src/metadata_scorer.py](#srcmetadata_scorerpy)
   - [src/feedback_handler.py](#srcfeedback_handlerpy)
   - [src/learning_engine.py](#srclearning_enginepy)
   - [src/search_cache.py](#srcsearch_cachepy)
   - [search_config.py](#search_configpy)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Database Schema](#10-database-schema)
11. [Scoring System Deep Dive](#11-scoring-system-deep-dive)
12. [Duplicate Detection Internals](#12-duplicate-detection-internals)
13. [Face Recognition Pipeline](#13-face-recognition-pipeline)
14. [Adaptive Learning System](#14-adaptive-learning-system)
15. [Configuration Tuning Guide](#15-configuration-tuning-guide)
16. [Persistent Files on Disk](#16-persistent-files-on-disk)
17. [Known Limitations](#17-known-limitations)
18. [Extending the Project](#18-extending-the-project)

---

## 1. What This Project Does

**Photo Scanner** is a fully **offline, local-first** AI-powered photo management CLI tool. It lets you:

- **Scan** a directory of images, extracting embeddings, OCR text, EXIF metadata, visual attributes (color, blur, exposure, scene type) and face data — storing everything in a local SQLite database.
- **Search** your photos with natural language queries (e.g. `"sunset at the beach"`, `"photo taken in Delhi"`, `"screenshot with error message"`) using a combination of CLIP visual embeddings, OCR text matching, and EXIF metadata scoring.
- **Find & manage duplicates** using perceptual hashing (pHash) and CLIP cosine similarity.
- **Cluster faces** across your entire library into identities (persons), then label and search by person name.
- **Continuously improve** search quality through a feedback loop — clicking, liking, or disliking results trains the system to reweight visual/OCR/metadata signals automatically.

No cloud. No API keys (except an optional Hugging Face token for the first CLIP model download). Everything runs on your machine.

---

## 2. Development Journey

This section chronicles every major design decision, problem encountered, and pivot made — so that anyone reading this years later can understand not just *what* the system does, but *why* it was built this way.

---

### Phase 1 — Finding the Right Embedding Model

The first challenge was: **how do you make a computer understand what a photo "contains" so a text query can find it?**

The initial research explored several approaches — traditional image feature extractors (ORB, SIFT), classification models (ResNet, MobileNet), and finally landed on **CLIP** (Contrastive Language–Image Pretraining), which was the perfect fit because it embeds *both* text and images into the same vector space. A query like `"dog at the beach"` and a matching photo end up with similar vectors — no labels, no categories needed.

The first CLIP integration used the full `openai/clip-vit-base-patch32` model loaded directly via HuggingFace Transformers. It was functionally correct but the combined model weight was too heavy (~600MB+ in memory), making it impractical for repeated use on a local machine.

**Solution:** Switched to **`sentence-transformers`** with the `clip-ViT-B-32` checkpoint. This wraps CLIP in a much more ergonomic API, handles batching cleanly, and the library itself is well maintained. The class was named `MobileCLIPHandler` — a naming artifact from an early experiment with a TFLite MobileCLIP variant that was later abandoned. The name stayed.

Additionally, a model quantization pass was explored to reduce the on-disk size of the weights. While quantization worked to reduce file size, the accuracy trade-off at lower bit depths wasn't worth it for this use case. Instead the approach shifted to simply caching the model in memory once loaded per process and using smart batching to amortize the startup cost.

---

### Phase 2 — Adding OCR for Text-in-Image Search

Photos of whiteboards, receipts, screenshots, and documents are a huge part of personal libraries but CLIP alone struggles with them — it sees the *visual style* of a document but not the specific words written on it.

The first OCR library integrated was **EasyOCR**. It was lightweight, had a simple Python API, and got reasonable results. It ran during scan and stored extracted text into a new `ocr_text` column in the database.

---

### Phase 3 — Rich Metadata Extraction

While OCR handled text, a huge untapped signal was the **EXIF metadata** embedded in every photo taken by a phone or camera. This was added to the scan pipeline via PIL's `_getexif()`:

- Image dimensions (`width`, `height`)
- Camera make and model (`device_make`, `device_model`)
- Date and time the photo was taken (`date_taken`)
- Camera settings: ISO, aperture, shutter speed, focal length, flash
- Orientation
- **GPS coordinates** (`gps_latitude`, `gps_longitude`, `gps_altitude`)

The GPS data opened up a powerful search use case — finding photos by location. However, raw GPS data is stored in the EXIF as degrees/minutes/seconds tuples and latitude/longitude floats — not human-readable city names.

**The GPS reverse geocoding problem:** Converting `(28.6139, 77.2090)` into `"New Delhi, Delhi, India"` normally requires a geocoding API (Google Maps, OpenCage, etc.). Using an external API would break the core guarantee of the project — fully offline, no external dependencies, no privacy leaks of your photo locations.

**Solution:** The `reverse_geocoder` Python library ships with an embedded offline dataset of ~200,000+ cities worldwide. It resolves coordinates to the nearest known city/town/state/country entirely from local data, with no network calls. The precision is approximate (nearest administrative region centroid, not street-level), but for the purpose of `search "photos from India"` or `search "photos from Delhi"`, it is more than sufficient. Precise GPS-to-address lookup would require an API and that defeats the purpose.

---

### Phase 4 — The OCR Bias Problem & Query Intent System

With OCR text and metadata both contributing to the search score, a serious ranking problem emerged: **screenshots and document photos were being ranked above visually relevant photos for almost any query**, because their OCR score was disproportionately high.

For example, a search for `"dog"` might surface a chat screenshot mentioning "hot dog" above an actual photo of a dog, simply because the OCR text score dominated the final score.

The root cause was a fixed weighting formula: `final = visual + ocr + metadata` with equal or near-equal weights regardless of what the user was actually searching for.

**Solution — Query Intent Classification system (`QueryAnalyzer`):**

The idea: before scoring, *classify what kind of search the user is doing*, then set the scoring weights dynamically.

Four intent categories were defined:

| Intent | When to use | Visual weight | OCR weight | Metadata weight |
|---|---|---|---|---|
| `VISUAL` | "dog playing in snow" | 0.8 | 0.1 | 0.1 |
| `METADATA` | "photos from Delhi in 2023" | 0.2 | 0.1 | 0.7 |
| `TEXT` | "screenshot with error message" | 0.2 | 0.7 | 0.1 |
| `HYBRID` | "Samsung photo at the beach" | ~0.5 | ~0.3 | ~0.2 |

Classification works by tokenizing the query and counting keyword matches against three curated keyword sets: `METADATA_KEYWORDS` (location names, date words, device names, camera terms), `VISUAL_KEYWORDS` (objects, scenes, colors), and `TEXT_KEYWORDS` (screenshot, receipt, note, etc.). Regex patterns additionally detect 4-digit years and dimension patterns like `1920x1080` as metadata signals.

For hybrid queries, the weight split is *proportional* to the keyword match counts — not just a binary toggle. This was a key insight that made hybrid queries feel natural.

The system eliminated the OCR dominance problem completely.

---

### Phase 5 — FAISS Vector Store

With hundreds or thousands of photos in the database, the original search approach of loading all embeddings into RAM on every search was becoming slow and memory-intensive.

**Solution:** Integrated **FAISS** (Facebook AI Similarity Search) as a persistent vector index. FAISS uses `IndexFlatIP` (inner product = cosine similarity on L2-normalized vectors) and serializes to a binary file (`faiss_index.bin`), paired with a numpy array (`faiss_id_map.npy`) that maps FAISS internal positions back to SQLite row IDs.

At search time, only the query embedding is computed; FAISS retrieves the top-100 candidates in milliseconds from the entire indexed library, and only those 100 records are fetched from SQLite. This made search effectively O(log n) in the number of photos.

The FAISS index is rebuilt after every scan and loaded at search time (or rebuilt if the index size doesn't match the DB record count).

---

### Phase 6 — EasyOCR → PaddleOCR (Text Quality Problem)

After real-world use, a significant quality problem with EasyOCR was discovered: **the extracted text was often garbled, misread, or incomplete** — especially for screenshots, low-contrast text, or images with mixed backgrounds. The `ocr_text` column in the database contained strings like `"Hel|o W0r|d"` instead of `"Hello World"`, making OCR-based search unreliable.

**Solution:** Migrated to **PaddleOCR** with the PP-OCRv3 model. PaddleOCR uses a detection model (finds text regions) followed by a recognition model (reads text), and its PP-OCRv3 checkpoint has significantly better accuracy on real-world photos compared to EasyOCR, particularly for:
- Mixed-background images
- Angled or curved text
- Non-standard fonts

The OCR output quality improved dramatically. The `OCRHandler` class was rewritten around PaddleOCR's API.

**Windows/CPU crash fix:** PaddleOCR on Windows with CPU-only setups would crash due to MKL-DNN (Intel's deep learning acceleration layer) initialization failures. This was fixed by disabling MKL-DNN via environment variables and monkey-patching `paddle.inference.Config.__init__` to call `self.disable_mkldnn()` on every config object created.

---

### Phase 7 — The Scan Time Crisis & OCR Optimization

Switching to PaddleOCR improved quality but created a new problem: **scan time exploded**. Processing 40 images took over **200 seconds** — far too slow for any practical use.

Profiling revealed OCR was the bottleneck by a large margin. Several optimization passes were applied:

**Round 1 — Batching & Preprocessing:**
- Switched from processing images one-by-one to batch recognition (`rec_batch_num=16`).
- Added a **preprocessing pipeline** before OCR: downscale the image to max 640px on the longest side (INTER_AREA interpolation), convert to grayscale, normalize pixel range. This reduces the pixel data fed into OCR by up to 10x on high-res phone photos.
- Result: ~200s → ~160s. Still too slow.

**Round 2 — OCR Gating (Document Detection):**
- The biggest win: realized that the vast majority of photos (outdoor scenes, portraits, food, etc.) contain *no meaningful text* and running OCR on them is pure wasted CPU.
- Added a **gatekeeper check**: at scan time, compute the CLIP embedding of each image and compare its cosine similarity to a fixed "document archetype" embedding (the text `"text document invoice receipt book page letter contract"`). Only images scoring above a `0.12` similarity threshold get OCR run on them.
- This eliminated OCR for ~85% of typical photo libraries.
- Result: ~160s → ~40s for real-world mixed photo sets. A 5x improvement.

The threshold `0.12` was tuned empirically — low enough to catch faint text in screenshots, high enough to skip purely visual photos.

---

### Phase 8 — Feedback System & Adaptive Learning

Even with excellent recall, individual searches would sometimes surface irrelevant results that consistently showed up for a particular query. There was no mechanism for the system to *learn* from user behavior.

**Solution — Feedback database (`feedback.db`):**

After each search, an interactive REPL allows the user to:
- Open any result image (number key) — recorded as `CLICKED`  
- Mark it as helpful (`y`) — recorded as `POSITIVE`
- Mark it as not helpful (`n`) — recorded as `NEGATIVE`

Every interaction is stored in `search_feedback` along with the weights that were active at search time and the per-component scores.

**Two-layer learning:**

1. **Per-result penalty/boost** (`FeedbackHandler.get_result_penalty`): A result's score is multiplied by a penalty factor derived from its specific feedback history for that query. Repeated negative feedback pushes a result toward 10% of its original score; repeated positive feedback boosts it by up to 30%.

2. **Weight preset adaptation** (`LearningEngine`): Across many interactions, the `LearningEngine` computes which scoring weights led to successful searches for each intent type and blends them into the active presets using Exponential Moving Average (EMA) with a learning rate of 0.25.

---

### Phase 9 — Duplicate Image Detection

Scanning a large photo library revealed a common problem: the same photo saved in multiple locations, or the same scene shot slightly differently, leading to cluttered results and wasted storage.

**Solution — Two-tier duplicate detection (`dedupe` command):**

**Tier 1 — Perceptual Hash (pHash):**
Every image gets a 256-bit pHash computed during scan. The algorithm: greyscale → resize 16×16 → 2D DCT (implemented in pure numpy, no scipy) → compare each DCT coefficient to the median → encode as bits.

Two images are compared by their **Hamming distance** (number of differing bits):
- Hamming = 0 → **EXACT** duplicate (pixel-identical hash, same image)
- Hamming ≤ 10 → **NEAR-EXACT** (JPEG re-saves, minor brightness edits, slight crops)
- Hamming > 10 → not a pHash match

To avoid O(n²) all-pairs comparison, pHashes are indexed in a **BK-Tree** (Burkhard-Keller Tree). BK-Trees use the metric space property of Hamming distance for O(log n) average range queries.

Groups are assembled using a **Union-Find** data structure (path-compressed, ranked) for near-linear-time grouping.

**Tier 2 — CLIP Embedding Similarity:**
Images that don't match via pHash are checked against each other using their CLIP embeddings. If cosine similarity ≥ 0.90 (configurable), they are grouped as **SEMANTIC** duplicates — visually similar photos that might be the same scene from a slightly different angle, or with a color filter applied.

Within each group, the **largest file by bytes** is kept (assumed highest quality). The `dedupe` command supports interactive review, auto-mark, or permanent disk deletion.

---

### Phase 10 — Face Recognition & Person Grouping

The final major feature: automatically finding all photos of the same person across the entire library without any labeling by the user.

**Detection — InsightFace `buffalo_l`:**  
The InsightFace library's `buffalo_l` model runs two steps:
1. **RetinaFace** detects face bounding boxes and returns bounding box coordinates + detection confidence score.
2. **ArcFace R100** generates a 512-dimensional L2-normalized face embedding for each detected face.

The model also estimates approximate **age** and **gender** for each face. Only faces with detection confidence ≥ 0.5 and a valid non-zero embedding are stored.

All face data is stored in the `faces` table: bounding box, detection score, the 512-D embedding (as raw bytes), estimated age, and gender.

**Clustering — DBSCAN + Graph Merge:**  
Face clustering is run via the `group-faces` command on all face embeddings in the database:

1. Compute pairwise cosine distance matrix across all face embeddings.
2. Run **DBSCAN** clustering (`eps=0.45` cosine distance, `min_samples=2`). DBSCAN naturally handles noise (faces that don't belong to any group are labeled `-1`).
3. Run a **graph merge** step with a looser `chain_threshold=0.50`. Build a sparse adjacency graph of all faces that are within this looser threshold, then compute connected components. This handles **identity drift** — the same person photographed decades apart may not be directly close in embedding space, but a chain of intermediate photos connects them. The threshold accounts for the natural variation in face embeddings caused by aging, different lighting, and angles.
4. Components with fewer than 2 faces remain unassigned.

Each surviving cluster becomes a `person` record in the database. Faces are updated with their `person_id`. Users can then label persons with `name-person <id> <name>` and search for all photos of that person with `search-person <name>`.

**Incremental assignment:** During scan (not just `group-faces`), new faces are immediately compared against existing person centroids. If a face falls within `cosine_distance < 0.45` of a known centroid, it is assigned to that person on the spot — without needing a full recluster.

---

## 3. High-Level Architecture

```
User CLI command
      │
      ▼
  main.py  ──────────────────────────────────────────┐
      │                                               │
  ┌───▼────────┐  ┌──────────────┐  ┌─────────────┐  │
  │ PhotoScanner│  │ PhotoDatabase│  │  FAISSIndex  │  │
  │ (scanner.py)│  │(database.py) │  │(faiss_index) │  │
  └─────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
        │                 │                 │           │
  Batch │                 │ SQLite          │ Binary     │
  of    │                 │ photos.db       │ faiss_index.bin
  images│                 │                 │
        │                 │                 │
  ┌─────▼───────────────────────────────────▼──────┐   │
  │  Per-image analysis pipeline (parallel threads) │   │
  │                                                 │   │
  │  MobileCLIPHandler  ──► CLIP embedding (512-D)  │   │
  │  ImageAnalyzer      ──► YOLOv8 + color/quality  │   │
  │  OCRHandler         ──► PaddleOCR text           │   │
  │  MetadataExtractor  ──► EXIF / GPS / device      │   │
  │  FaceDetector       ──► InsightFace embeddings   │   │
  └─────────────────────────────────────────────────┘   │
                                                         │
  ┌──────────────── SEARCH PATH ────────────────────┐    │
  │ QueryAnalyzer ──► intent + dynamic weights       │◄───┘
  │ SearchCache   ──► text embedding disk cache      │
  │ FAISSIndex    ──► top-k ANN retrieval            │
  │ MetadataScorer──► EXIF/date/device scoring       │
  │ FeedbackHandler──► result penalty/boost          │
  │ LearningEngine──► adaptive weight updates        │
  └─────────────────────────────────────────────────┘
```

---

## 4. Repository Layout

```
Photo_scanner/
├── main.py                  # CLI entry point — all commands live here
├── search_config.py         # Central configuration class (SearchConfig)
├── src/
│   ├── __init__.py
│   ├── scanner.py           # PhotoScanner: orchestrates the scan pipeline
│   ├── database.py          # PhotoDatabase: all SQLite I/O
│   ├── model_handler.py     # MobileCLIPHandler: wraps SentenceTransformer CLIP
│   ├── faiss_index.py       # FAISSIndex: vector ANN index (build/load/search/save)
│   ├── image_analyzer.py    # ImageAnalyzer: YOLOv8 face count, scene, color, quality
│   ├── ocr_handler.py       # OCRHandler: PaddleOCR text extraction
│   ├── face_detector.py     # detect_faces(): InsightFace ArcFace detection
│   ├── person_clusterer.py  # cluster_faces(): DBSCAN + graph-merge clustering
│   ├── duplicate_detector.py# DuplicateDetector: pHash BK-tree + embedding FAISS
│   ├── query_analyzer.py    # QueryAnalyzer: classify intent → dynamic weights
│   ├── metadata_scorer.py   # score_batch_metadata(): EXIF attribute scoring
│   ├── feedback_handler.py  # FeedbackHandler: store/query feedback.db
│   ├── learning_engine.py   # LearningEngine: EMA weight updates from feedback
│   └── search_cache.py      # SearchCache: LRU + TTL text-embedding cache
├── assets/
│   └── local_clip_model/    # Optional: offline CLIP model directory
├── embedding_cache/         # Disk cache for serialized text embeddings
├── test_images/             # Sample images for manual testing
├── photos.db                # SQLite — photo index (auto-created on first scan)
├── feedback.db              # SQLite — user feedback log
├── faiss_index.bin          # FAISS index binary (auto-updated after scan)
├── faiss_id_map.npy         # NumPy array mapping FAISS positions → DB row IDs
├── yolov8n.pt               # YOLOv8 nano weights (downloaded by ultralytics)
├── search_results.log       # Last search result output (plain text)
├── pyproject.toml           # Poetry project definition + dependencies
├── inspect_db.py            # Debug helper: prints DB contents to stdout
└── inspect_model.py         # Debug helper: tests CLIP model embed/similarity
```

---

## 5. Dependency Stack

| Package | Purpose |
|---|---|
| `sentence-transformers` | CLIP model (`clip-ViT-B-32`) — 512-D visual+text embeddings |
| `faiss-cpu` | Approximate Nearest Neighbor (ANN) index for fast search |
| `ultralytics` | YOLOv8n — person/object detection during scan |
| `insightface` | RetinaFace + ArcFace — face detection & 512-D face embeddings |
| `paddleocr` + `paddlepaddle` | PP-OCRv3 — text extraction from images |
| `opencv-python` | Image processing (resize, HSV, Laplacian, Canny) |
| `pillow` | Image I/O (EXIF reading, PIL Image) |
| `numpy` | All vector math |
| `scipy` | DBSCAN (via sklearn), connected-components, cdist |
| `scikit-learn` | DBSCAN clustering for face grouping |
| `reverse-geocoder` | Offline GPS → city/country name resolution |
| `python-dotenv` | Load `HF_TOKEN` from `.env` for Hugging Face auth |
| `psutil` | Memory usage logging (`MobileCLIPHandler.log_memory`) |
| `tqdm` | Progress bars during scan |
| `tensorflow` / `onnxruntime` | Runtime backends for some embedded models |

Install all with:
```bash
poetry install
```

---

## 6. Setup & First Run

```bash
# 1. Clone + install
git clone <repo-url>
cd Photo_scanner
poetry install

# 2. (Optional) Create .env with Hugging Face token if you hit rate limits:
echo "HF_TOKEN=hf_your_token_here" > .env

# 3. First scan (downloads CLIP model ~400MB on first run)
poetry run python main.py scan "C:/Users/You/Pictures"

# 4. Search
poetry run python main.py search "beach sunset with family"

# 5. Find duplicates
poetry run python main.py dedupe

# 6. Group faces
poetry run python main.py group-faces
poetry run python main.py name-person 1 "Alice"
poetry run python main.py search-person "Alice"
```

**Offline mode:** Place a pre-downloaded SentenceTransformer model directory at `assets/local_clip_model/`. The scanner auto-detects it and skips any network call.

---

## 7. CLI Reference

### `scan <directory>`

Recursively finds all `.jpg/.jpeg/.png/.bmp/.gif/.webp` files under `<directory>`.

- **Incremental:** Only processes new or modified files (compares mtime + size against DB).
- After scanning, rebuilds the FAISS index.

```bash
python main.py scan "D:/Photos/Vacation2024"
```

---

### `search <query>`

Natural language search across your photo index.

```bash
python main.py search "invoice receipt from 2023"
python main.py search "photo taken with Samsung on a mountain"
python main.py search "dog playing in snow"
```

**How it works (simplified):**
1. `QueryAnalyzer` classifies the query intent (VISUAL / METADATA / TEXT / HYBRID) and sets dynamic weights.
2. CLIP embeds the query text → FAISS retrieves top-100 candidate images by cosine similarity.
3. Each candidate gets an OCR score (keyword matching) and a metadata score (EXIF attributes).
4. Final score = `visual * w_v + ocr * w_o + metadata * w_m` (weights set by intent).
5. Results are filtered using an adaptive threshold (gap detection + proportional floor).
6. Feedback penalties/boosts from previous interactions are applied.
7. Interactive mode lets you open images and mark results as helpful/not helpful.

---

### `dedupe [options]`

Find duplicate and near-duplicate images.

| Flag | Default | Meaning |
|---|---|---|
| `--phash-threshold N` | `10` | Max Hamming distance for pHash near-duplicate (0 = pixel-identical) |
| `--embedding-threshold F` | `0.90` | Min cosine similarity for semantic duplicate |
| `--no-embedding` | off | Skip CLIP embedding check (faster, pHash only) |
| `--auto-mark` | off | Mark all found duplicates in DB without asking |
| `--delete` | off | **Irreversibly** delete duplicate files from disk |
| `--list` | off | List all previously flagged duplicates from DB |

```bash
python main.py dedupe --phash-threshold 5 --auto-mark
python main.py dedupe --delete   # DANGEROUS — deletes files permanently
```

---

### `group-faces [--reset]`

Clusters all face embeddings currently in the database into person identity groups using DBSCAN + graph merging.

- `--reset` wipes all existing person assignments and reclusters from scratch.

```bash
python main.py group-faces
python main.py group-faces --reset
```

---

### `name-person <person_id> <name>`

Assigns a human-readable label to a person cluster.

```bash
python main.py name-person 3 "Mom"
```

---

### `search-person <name_or_id>`

Lists all photos containing the specified person.

```bash
python main.py search-person "Mom"
python main.py search-person 3
```

---

## 8. Module-by-Module Reference

### `main.py`

**Role:** Single CLI entry point. Uses `argparse` with sub-parsers for each command. Contains all command handler logic inline (not split into separate functions) — this is intentional for readability of the flow.

**Key design notes:**
- The `search` command loads `MobileCLIPHandler` *lazily* — if the text embedding for the query is already in `SearchCache`, it skips loading the model entirely (significant startup time saving).
- Feedback mode runs a `while True` REPL after results are shown, accepting numeric (open image), `y` (positive), `n` (negative), `s` (stats), `q` (quit) inputs.
- The `dedupe` command handles 3 modes: `--list`, `--auto-mark`/`--delete`, and interactive review, in priority order.

---

### `src/scanner.py`

**Class:** `PhotoScanner`

**`__init__(model_path, db_path="photos.db")`**  
Initializes: `PhotoDatabase`, `MobileCLIPHandler`, `OCRHandler`, `ImageAnalyzer`.  
Also precomputes a *document gatekeeper embedding* — a CLIP embedding of the string `"text document invoice receipt book page letter contract"`. During scanning, any image with cosine similarity > `0.12` to this embedding gets its OCR run; others skip OCR entirely (saves significant time on pure photos).

**`scan_directory(directory)`**  
Core scan loop:

1. Walks disk to build `{path: (mtime, size)}`.
2. Queries DB for existing paths+mtime → computes new/modified/deleted sets.
3. Removes deleted files from DB.
4. Processes new+modified files in **batches of 16**:
   - CLIP embeddings for the entire batch (GPU-friendly batching).
   - Parallel `ThreadPoolExecutor` for: metadata extraction, visual analysis, OCR (conditional on gatekeeper score).
   - Stores result in DB (`add_photo` or `update_photo`).
   - Computes pHash and stores it.
   - Detects faces and stores face embeddings.
5. Rebuilds FAISS index.

**`_process_faces_for_photo(path)`**  
Detects faces, stores each into the `faces` table, and incrementally assigns the best-matching existing `person_id` using centroid comparison. Centroids are refreshed once per scan (lazy, first face found triggers it).

**`_extract_metadata(image_path)`**  
Reads EXIF via PIL. Extracts: `device_make/model`, `date_taken`, `iso`, `aperture`, `shutter_speed`, `focal_length`, `flash`, `orientation`, `gps_latitude/longitude/altitude`, and resolves GPS → location name via `reverse_geocoder`.

---

### `src/database.py`

**Class:** `PhotoDatabase(db_path="photos.db")`

SQLite wrapper. Creates and auto-migrates the schema on init (adds `ocr_text`, `phash`, `is_duplicate`, `duplicate_of` columns if missing).

**Key methods:**

| Method | Purpose |
|---|---|
| `add_photo(path, size, mtime, embedding, metadata, ocr_text)` | INSERT new photo record |
| `update_photo(...)` | UPDATE existing record (re-scan) |
| `get_all_embeddings_with_ids()` | Returns `(ids, embeddings)` np arrays for FAISS build |
| `get_batch_by_ids(ids)` | Fetch path+ocr+metadata for a list of DB IDs (post-FAISS retrieval) |
| `get_scanned_paths_with_mtime()` | Returns `{path: (mtime, size)}` dict for incremental scan diffing |
| `update_phash(path, phash_bytes)` | Store 32-byte pHash blob |
| `get_all_for_dedup()` | Fetch all records with phash+embedding for dedupe |
| `mark_as_duplicate(path, original)` | Set `is_duplicate=1, duplicate_of=original` |
| `get_duplicates()` | List all flagged duplicates |
| `add_face(photo_id, bbox, det_score, embedding, age, gender)` | Insert face row, returns face_id |
| `get_all_faces_with_embeddings()` | All face rows with embedding arrays for clustering |
| `add_person(name, rep_face_id)` | Insert person row, returns person_id |
| `get_person_centroids()` | `{person_id: centroid_embedding}` for incremental assignment |
| `delete_all_persons()` | Wipe persons for full recluster |

**Embedding storage:** `float32` numpy arrays are stored as raw bytes (`embedding.astype(np.float32).tobytes()`). Read back with `np.frombuffer(blob, dtype=np.float32)`.

---

### `src/model_handler.py`

**Class:** `MobileCLIPHandler(model_name="clip-ViT-B-32")`

Thin wrapper around `SentenceTransformer`. Despite being called "MobileCLIP" (a naming artifact from an earlier TFLite approach), this actually uses the standard OpenAI CLIP ViT-B/32 via `sentence-transformers`.

- Checks `assets/local_clip_model/` first for an offline copy.
- Falls back to downloading from Hugging Face (uses `HF_TOKEN` env var if present).
- `get_image_embeddings_batch(paths)` → returns `(N, 512)` float32 array. Skipped/failed images get zero vectors.
- `get_text_embedding(text)` → `(512,)` float32. Falls back to zero vector on error.
- `log_memory(stage)` logs process RSS memory (useful for diagnosing OOM).

---

### `src/faiss_index.py`

**Class:** `FAISSIndex(index_dir=".")`

Wraps a FAISS `IndexFlatIP` (inner product = cosine similarity on L2-normalized vectors).

**Files on disk:**
- `faiss_index.bin` — the FAISS index binary
- `faiss_id_map.npy` — 1-D int64 numpy array mapping FAISS position → SQLite row ID

**`load_or_build(db)`**  
Tries to load from disk. If the index size doesn't match DB count, rebuilds. This is called at search time and after every scan.

**`search(query_embedding, top_k=100)`**  
Returns `(db_ids, scores)` — database IDs and cosine similarity scores for the top-k matches.

---

### `src/image_analyzer.py`

**Class:** `ImageAnalyzer`

Runs 4 analyses per image during scan:

1. **`_detect_faces(path)`** — Uses **YOLOv8n** (`yolov8n.pt`, COCO class 0 = person) to count people in the image. Sets `face_count`, `has_faces`, `face_category` (`no_faces`/`portrait`/`duo`/`group`).

2. **`_classify_scene(img_array)`** — Heuristic using OpenCV HSV histograms:
   - Sky-blue ratio in the top-third of image → outdoor
   - Green vegetation ratio → natural outdoor
   - Edge density (Canny) → urban
   - Default: indoor

3. **`_analyze_colors(img_array)`** — HSV histogram binning per 12-color HSV range → `dominant_colors` list. Warm vs cool pixel ratio → `color_tone`. Mean saturation → `color_vibrance`.

4. **`_score_quality(img_array)`** — Laplacian variance for blur (`is_blurry` = variance < 100). Brightness histogram for exposure classification. Combined point system → `quality_rating` (poor/fair/good/excellent).

All outputs are merged into the photo's `metadata` JSON column in the DB.

---

### `src/ocr_handler.py`

**Class:** `OCRHandler(languages=['en'], min_confidence=0.5)`

Uses **PaddleOCR PP-OCRv3** for text extraction.

**Gated execution:** OCR is only run if the image's CLIP embedding is similar enough to a document archetype (threshold checked in `PhotoScanner`). This prevents OCR on every vacation photo.

**Preprocessing pipeline (`_preprocess_for_ocr`):**
1. Load via cv2
2. Downscale longest side to 640px (INTER_AREA)
3. Convert to grayscale
4. Normalize to 0–255 range
5. Pass to PaddleOCR (det → rec pipeline)

Results are filtered by `min_confidence=0.5` and joined into a single string stored in `ocr_text`.

**MKL-DNN is disabled** via environment flags and a monkey-patch on `paddle.inference.Config.__init__` — this avoids a known crash on Windows/CPU-only setups.

---

### `src/face_detector.py`

**Function:** `detect_faces(image_path) → List[dict]`

Uses **InsightFace** `buffalo_l` model (RetinaFace detector + ArcFace R100 recognizer). Downloaded automatically to `~/.insightface/models/` on first run.

Each returned face dict:
```python
{
    "bbox": [x1, y1, x2, y2],   # float pixels
    "det_score": float,           # detection confidence [0-1]
    "embedding": np.ndarray,      # 512-D ArcFace, L2-normalised
    "age":   int or None,
    "gender": "M" | "F" | None,
}
```

Only faces with `det_score >= 0.5` and a valid non-zero embedding are returned. The model is lazily loaded (singleton `_face_app`) so the ~2 second init cost only happens once per scan run.

---

### `src/person_clusterer.py`

**Functions:** `cluster_faces(embeddings, face_ids)`, `assign_new_face(embedding, centroids)`

**Clustering algorithm (used by `group-faces` command):**

1. Build pairwise cosine distance matrix (`O(N²)` — acceptable since faces are a small fraction of total images).
2. **DBSCAN** with `eps=0.45` (cosine distance), `min_samples=2`. Labels noise as `-1`.
3. **Graph-merge** with a looser `CHAIN_THRESHOLD=0.50` — builds a sparse adjacency graph and finds connected components. This handles *identity drift* (you at age 10 vs age 30 — DBSCAN might split them, but the graph merge chains them via intermediate ages).
4. Components with < `MIN_CLUSTER_SIZE=2` faces remain unassigned.

**Incremental assignment (`assign_new_face`):** Used during scan to assign newly detected faces to existing person clusters without full recluster. Finds the closest centroid; assigns if cosine distance < `COSINE_THRESHOLD=0.45`.

---

### `src/duplicate_detector.py`

See [Section 11](#11-duplicate-detection-internals) for the full algorithm breakdown.

**Key classes/functions:**

| Item | Purpose |
|---|---|
| `compute_phash(path, hash_size=16)` | 256-bit pHash via 2D DCT. No scipy needed. |
| `hamming_distance(a, b)` | XOR bit-count |
| `phash_to_bytes(h)` / `bytes_to_phash(b)` | Serialize/deserialize for SQLite BLOB |
| `BKTree` | Burkhard-Keller tree for sub-linear pHash range search |
| `UnionFind` | Path-compressed union-find for group membership |
| `DuplicateDetector` | Orchestrates pHash + embedding duplicate detection |

---

### `src/query_analyzer.py`

**Class:** `QueryAnalyzer`

Classifies queries into 4 intents:

| Intent | Example query | Preset weights |
|---|---|---|
| `VISUAL` | `"dog playing in snow"` | V=0.8, O=0.1, M=0.1 |
| `METADATA` | `"photos from Delhi in 2023"` | V=0.2, O=0.1, M=0.7 |
| `TEXT` | `"screenshot with error message"` | V=0.2, O=0.7, M=0.1 |
| `HYBRID` | `"Samsung photo at the beach"` | V≈0.5, O≈0.3, M≈0.2 |

**Classification:** Tokenizes query, counts matches in `METADATA_KEYWORDS`, `VISUAL_KEYWORDS`, `TEXT_KEYWORDS`. Regex patterns additionally catch 4-digit years (`\b(19|20)\d{2}\b`) and dimension patterns (`\d+x\d+`) as metadata signals.

**Hybrid weighting** is proportional — if 3 of 5 matched tokens are visual, visual weight gets `0.4 + (3/5 * 0.3) = 0.58`.

**Adaptive weights:** On init, `QueryAnalyzer` optionally loads learned weight overrides from `LearningEngine` (if `ENABLE_ADAPTIVE_LEARNING=True` and enough feedback exists).

---

### `src/metadata_scorer.py`

**Function:** `score_batch_metadata(query, metadata_list) → (scores, reasons)`

Scores each photo's metadata against the query. Looks for matches in:
- Date fields (`date_taken`, `date_modified`) — matches year, month name, day
- Location fields (`location_name`, `location`) — substring match
- Device fields (`device`, `device_make`, `device_model`) — substring match
- Camera settings (`iso`, `aperture`, `focal_length`, `flash`)
- Visual attributes (`face_category`, `scene_type`, `scene_environment`, `dominant_colors`, `color_tone`, `color_vibrance`, `exposure`, `quality_rating`, `is_blurry`)
- Dimension terms (`width`, `height`, orientation keywords)
- GPS altitude

Returns a float score (additive, typically 0–3+) and a list of human-readable `reasons` strings explaining what matched.

---

### `src/feedback_handler.py`

**Class:** `FeedbackHandler(db_path="feedback.db")`

SQLite-backed feedback store. Schema: `search_feedback` table with columns: `query`, `query_intent`, `weights_used` (JSON), `result_path`, `result_rank`, `result_scores` (JSON), `feedback_type`, `timestamp`.

**Feedback types (`FeedbackType` enum):**
- `CLICKED` — user opened the image
- `POSITIVE` — user marked as helpful (`y`)
- `NEGATIVE` — user marked as not helpful (`n`)
- `IGNORED` — (reserved, not currently emitted)

**`get_result_penalty(result_path, query)`**  
Returns a multiplier applied to a result's score:
- Pure negative: `max(0.1, 1.0 - negatives * 0.12)` (minimum 10% of original score)
- Mixed: `max(0.3, 1.0 - (neg_ratio * 0.7))`
- Pure positive: `min(1.3, 1.0 + positives * 0.06)` (maximum 30% boost)
- No feedback: `1.0`

---

### `src/learning_engine.py`

**Class:** `LearningEngine`

Uses **Exponential Moving Average (EMA)** to update weight presets based on successful search interactions.

**`update_weights(current_presets)`**  
For each intent type:
1. Fetches aggregated successful-interaction weights from `FeedbackHandler.get_learning_data()`.
2. Requires `>= min_samples` (default 5) and `>= min_success_rate` (default 20%).
3. `new_weight = current * (1 - lr) + observed * lr` with lr=0.25.
4. Clamps change to `max_adjustment=0.15` per update cycle.

---

### `src/search_cache.py`

**Class:** `SearchCache(max_entries=50, ttl_seconds=300)`

In-memory LRU cache + disk persistence for text embeddings. Avoids reloading the CLIP model for repeated queries.

- Cache key: SHA-256 hash of the query string
- Disk storage: `embedding_cache/` directory, one `.npy` file per cached embedding
- TTL: 5 minutes by default; expired entries are evicted on access

---

### `search_config.py`

**Class:** `SearchConfig` (class-level constants, not instantiated)

Single source of truth for all tuneable parameters. See [Section 14](#14-configuration-tuning-guide) for guidance.

**`SearchConfig.get_config()`** returns a dict of all parameters — this is what all modules read at runtime, so changing a class attribute instantly affects all subsequent searches.

---

## 9. Data Flow Diagrams

### Scan Flow

```
scan_directory(dir)
        │
        ├─► Walk disk → {path: (mtime, size)}
        ├─► DB.get_scanned_paths_with_mtime() → existing records
        │
        ├─► new_files = paths not in DB
        ├─► modified_files = paths with changed mtime or size
        ├─► deleted_files = DB paths not on disk → DB.remove_photos()
        │
        └─► For each batch of 16 from (new + modified):
                │
                ├─[Thread 1]─► MobileCLIPHandler.get_image_embeddings_batch()
                ├─[Thread 2]─► scanner._extract_metadata() (EXIF via PIL)
                ├─[Thread 3]─► ImageAnalyzer.analyze() (YOLO + color + quality)
                └─[Thread 4]─► OCRHandler.extract_text()  ← only if doc-similar
                        │
                        ▼
                DB.add_photo() or DB.update_photo()
                DB.update_phash(compute_phash())
                FaceDetector.detect_faces() → DB.add_face() → assign_new_face()
                        │
                        ▼
                FAISSIndex.build_index() + save()
```

### Search Flow

```
search(query)
        │
        ├─► QueryAnalyzer.analyze_query() → (intent, weights, debug_info)
        ├─► QueryAnalyzer.get_ocr_tokens() → meaningful tokens
        │
        ├─► SearchCache.get_text_embedding(query)
        │        ├─ HIT  → skip model load, use cached embedding
        │        └─ MISS → MobileCLIPHandler.get_text_embedding()
        │                   → normalize → SearchCache.set_text_embedding()
        │
        ├─► FAISSIndex.load_or_build(db)
        │         → search(text_emb, top_k=100) → (candidate_ids, vector_scores)
        │
        ├─► DB.get_batch_by_ids(candidate_ids) → candidate records
        │
        └─► For each candidate:
                v_score = vector_scores[i]
                o_score = OCR token matching (exact / partial / fuzzy)
                m_score, reasons = score_batch_metadata(query, [metadata])
                final = v_score*weights.visual + o_score*weights.ocr + m_score*weights.meta
                │
                ▼
        Sort by final score DESC
        Apply FeedbackHandler.get_result_penalty() to each result
        Re-sort
        Apply adaptive threshold filtering (gap + proportional floor)
        Print + log results
        Enter feedback REPL
```

---

## 10. Database Schema

### `photos` table (`photos.db`)

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment row ID (used by FAISS id_map) |
| `path` | TEXT UNIQUE | Absolute path to the image file |
| `filename` | TEXT | `os.path.basename(path)` |
| `size_bytes` | INTEGER | File size at scan time |
| `modified_time` | REAL | `os.stat().st_mtime` |
| `embedding` | BLOB | 512-D float32 CLIP embedding, raw bytes |
| `metadata` | TEXT | JSON dict of EXIF + visual attributes |
| `ocr_text` | TEXT | Extracted text (empty string if no text detected) |
| `phash` | BLOB | 32-byte big-endian pHash (256-bit integer) |
| `is_duplicate` | INTEGER | 0 = original, 1 = flagged duplicate |
| `duplicate_of` | TEXT | Path of the "keep" image (if is_duplicate=1) |

### `faces` table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Face row ID |
| `photo_id` | INTEGER FK | Reference to `photos.id` |
| `person_id` | INTEGER FK | Reference to `persons.id` (NULL if unassigned) |
| `bbox_x1/y1/x2/y2` | REAL | Bounding box in pixel coordinates |
| `det_score` | REAL | InsightFace detection confidence [0-1] |
| `embedding` | BLOB | 512-D ArcFace float32, raw bytes |
| `age_estimate` | INTEGER | Estimated age (or NULL) |
| `gender` | TEXT | "M", "F", or NULL |

### `persons` table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Person cluster ID |
| `name` | TEXT | Human-assigned name (NULL until `name-person` is run) |
| `representative_face_id` | INTEGER | FK to `faces.id` — the "best" face for this person |
| `created_at` / `updated_at` | REAL | Unix timestamps |

### `search_feedback` table (`feedback.db`)

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | — |
| `query` | TEXT | The search query |
| `query_intent` | TEXT | VISUAL/METADATA/TEXT/HYBRID |
| `weights_used` | TEXT | JSON `{visual, ocr, metadata}` |
| `result_path` | TEXT | Absolute path of the result image |
| `result_rank` | INTEGER | Position in results (1-indexed) |
| `result_scores` | TEXT | JSON `{visual, ocr, metadata, final}` |
| `feedback_type` | TEXT | POSITIVE/NEGATIVE/CLICKED/IGNORED |
| `timestamp` | DATETIME | Auto-set by SQLite |

---

## 11. Scoring System Deep Dive

### Final Score Formula

```
final_score = (visual_score * w_visual)
            + (ocr_score    * w_ocr)
            + (metadata_score * w_metadata)
```

### Visual Score (`v_score`)

Raw cosine similarity from FAISS inner product search on L2-normalized CLIP embeddings. Range: typically `[0.10, 0.45]` for general photos. A score of `0.3+` is usually a good visual match.

### OCR Score (`o_score`)

Token-level matching between query tokens and `ocr_text`. Scored additively:
- Exact full-word match: `+1.0`
- Substring match: `+0.6`
- Fuzzy neighbor (length difference ≤ 2): `+0.3`

Capped at `3.0`. Only tokens ≥ `OCR_MIN_TOKEN_LENGTH=3` chars are considered.

### Metadata Score (`m_score`)

Additive score from `metadata_scorer.py`. Typical contributions:
- Year match in date: `+0.5`
- Month match: `+0.3`
- Location name match: `+0.5`
- Device brand match: `+0.4`
- Scene type match (indoor/outdoor): `+0.3`

### Adaptive Threshold Filtering

After sorting, results are filtered using this 3-way formula:

```python
relative_cut = top_score * (1 - RELATIVE_THRESHOLD)   # e.g. top=0.3, rel=0.30 → cut=0.21
floor_cut    = top_score * FLOOR_RATIO                 # e.g. top=0.3, floor=0.70 → cut=0.21
threshold    = max(relative_cut, floor_cut)            # tightest of the two

# Gap detection: if a sudden score drop > 10% of top_score is found in top-30,
# the gap threshold is also considered (takes max with proportional threshold).
```

This ensures that even visually weak queries (low absolute scores) still yield relative near-matches, without being blocked by a hard minimum.

---

## 12. Duplicate Detection Internals

### Phase 1: Perceptual Hash (pHash)

1. **Compute pHash:** Image → greyscale → resize to 16×16 → 2D DCT (hand-rolled in numpy, no scipy) → take 255 lowest-frequency components (excluding DC) → bits = `dct_low > median(dct_low)` → pack into 256-bit integer.

2. **BK-Tree indexing:** All pHashes are inserted into a BK-Tree (Burkhard-Keller Tree) using Hamming distance as the metric. This enables **O(log n)** range queries instead of O(n²) all-pairs comparisons.

3. **Union-Find grouping:** For each image, query the BK-Tree for neighbours within `phash_threshold` Hamming distance. Union all found pairs.

4. **Classifying:**
   - Hamming distance = 0 → `exact` duplicate
   - 0 < distance ≤ `phash_threshold` → `near_exact`

### Phase 2: CLIP Embedding Similarity

Run only on images **not already covered** by pHash groups (avoids double-counting).

For N ≥ 1000 images: uses FAISS `IndexIVFFlat` with `nlist = min(√N, 256)` and `nprobe = nlist//4`.
For N < 1000: uses `IndexFlatIP` (exact).

Each image queries its k=16 nearest neighbours. Pairs with cosine similarity ≥ `embedding_threshold` are unioned.

Grouped images from this phase are marked `semantic`.

### Representative Selection

Within each duplicate group, the **largest file** (by bytes) is kept as the "original" — on the assumption that larger = higher quality. Ties broken lexicographically for reproducibility.

---

## 13. Face Recognition Pipeline

```
Photo scan
    │
    ▼
face_detector.detect_faces(path)
    ├─ InsightFace buffalo_l
    │   ├─ RetinaFace: bounding box detection
    │   └─ ArcFace R100: 512-D L2-normalised embedding
    └─ Returns: [{bbox, det_score, embedding, age, gender}]
           (only det_score >= 0.5)
    │
    ▼
DB.add_face(photo_id, bbox, det_score, embedding, age, gender)
    │
    ├─ Incremental assignment:
    │   person_clusterer.assign_new_face(embedding, person_centroids)
    │   └─ Nearest centroid with cosine dist < 0.45 → assign person_id
    │
    ▼
group-faces command (run manually):
    │
    ├─ Load all face embeddings from DB
    ├─ cluster_faces(embeddings, face_ids)
    │   ├─ Pairwise cosine distance matrix
    │   ├─ DBSCAN (eps=0.45, min_samples=2)
    │   └─ Graph-merge (chain_threshold=0.50)
    │       └─ connected_components (scipy.sparse)
    │
    ├─ Create person records in DB for each cluster
    └─ Update face rows with person_id
```

---

## 14. Adaptive Learning System

The system learns which scoring weights work best for each query intent type.

**Signal collection** (happens automatically during `search`):
- Every time a user opens an image (CLICKED), marks as helpful (POSITIVE), or marks as not helpful (NEGATIVE), `FeedbackHandler.record_feedback()` is called, storing the query, intent, weights used at search time, and the individual scores.

**Weight update** (at search time, before results are shown):
- `QueryAnalyzer._load_learned_weights()` calls `LearningEngine.update_weights()`.
- For each intent type with ≥ 5 successful interactions and ≥ 20% success rate:
  - Computes the average weights that led to successful interactions.
  - EMA-blends: `new = current * 0.75 + observed * 0.25`.
  - Clamps to ±0.15 change per update.
- Result: over time, if users consistently click images with high metadata scores for "location" queries, the METADATA weight for METADATA-intent queries increases automatically.

**Penalty/boost system** (per-result, per-query):
- Independently from weight learning, each result gets a per-result multiplier based on its specific feedback history for that query.

---

## 15. Configuration Tuning Guide

All tuning is done in `search_config.py` by editing class-level constants.

### Search Strictness

| Want | Change |
|---|---|
| Fewer, higher-confidence results | Decrease `RELATIVE_THRESHOLD` (e.g. `0.15`), increase `FLOOR_RATIO` (e.g. `0.85`) |
| More lenient results | Increase `RELATIVE_THRESHOLD` (e.g. `0.40`), decrease `FLOOR_RATIO` (e.g. `0.60`) |
| Hard cap on result count | Decrease `MAX_RESULTS` |

### Duplicate Detection

| Want | Change |
|---|---|
| Only pixel-identical duplicates | `DEDUP_PHASH_THRESHOLD = 0` |
| Near-exact (JPEG re-saves, minor crops) | `DEDUP_PHASH_THRESHOLD = 10` (default) |
| Looser near-duplicates | `DEDUP_PHASH_THRESHOLD = 20` |
| Semantic duplicates too | `DEDUP_USE_EMBEDDING = True`, `DEDUP_EMBEDDING_THRESHOLD = 0.95` |
| More semantic sensitivity | Lower `DEDUP_EMBEDDING_THRESHOLD` (e.g. `0.90`) |

### Adaptive Learning

| Want | Change |
|---|---|
| Disable adaptive learning | `ENABLE_ADAPTIVE_LEARNING = False` |
| Learn faster (fewer samples needed) | Decrease `MIN_FEEDBACK_SAMPLES` |
| More aggressive weight updates | Increase `LEARNING_RATE`, `MAX_WEIGHT_ADJUSTMENT` |
| Disable per-result penalties | `ENABLE_RESULT_PENALTIES = False` |

### OCR Sensitivity

| Want | Change |
|---|---|
| Match shorter words | Decrease `OCR_MIN_TOKEN_LENGTH` (e.g. `2`) |
| More conservative partial matches | Decrease `OCR_PARTIAL_MATCH_SCORE` |

---

## 16. Persistent Files on Disk

| File | Created by | Safe to delete? |
|---|---|---|
| `photos.db` | First scan | ⚠ Deletes entire photo index, need full rescan |
| `feedback.db` | First search with feedback | ✅ Loses learned preferences only |
| `faiss_index.bin` | After each scan | ✅ Rebuilt on next search |
| `faiss_id_map.npy` | After each scan | ✅ Rebuilt on next search |
| `embedding_cache/` | First search query | ✅ Loses text embedding cache |
| `search_results.log` | Each search | ✅ Just the last search output |
| `yolov8n.pt` | First scan (ultralytics auto-download) | ✅ Re-downloaded on next scan |
| `~/.insightface/models/buffalo_l/` | First face detection | ✅ Re-downloaded on next scan |

---

## 17. Known Limitations

1. **No GPU support for OCR/Face detection** — both use CPU (`CPUExecutionProvider`). Swap to `CUDAExecutionProvider` in `face_detector.py` if a CUDA GPU is available for significant speedup.

2. **pHash robustness** — pHash is robust to minor JPEG artifacts, slight brightness changes, and small crops. It is **not** robust to: heavy crops, rotations > ~5°, color filters, or significant resizing. Semantic duplicates of that nature are caught by CLIP embedding pass.

3. **OCR gatekeeper false negatives** — The document gatekeeper (cosine similarity to a text-archetype embedding) may miss screenshots or images with minimal text. Lower `doc_threshold` in `PhotoScanner.__init__` (currently `0.12`) to be more inclusive.

4. **Face clustering needs `group-faces` command** — Incremental assignment during scan only assigns faces that closely match existing person centroids. New people won't be clustered until `group-faces` is run explicitly.

5. **Language support in OCR** — Currently configured for English (`lang='en'`). Change in `OCRHandler.__init__` or extend `LANGUAGE_MAP` for multi-language support.

6. **Search cache size** — `SearchCache` defaults to 50 entries, 5-minute TTL. For heavy use, increase these in `main.py` at `SearchCache(max_entries=50, ttl_seconds=300)`.

---

## 18. Extending the Project

### Add a New CLI Command

1. Add a new sub-parser in `main.py` under the `subparsers` block.
2. Add an `elif args.command == "your-command":` handler block.

### Add a New Metadata Field

1. Extract the field in `PhotoScanner._extract_metadata()` or `ImageAnalyzer.analyze()`.
2. Add a scoring rule in `src/metadata_scorer.py` in the appropriate section.
3. Add relevant keywords to `QueryAnalyzer.METADATA_KEYWORDS` if applicable.

### Add a New Supported Image Format

Add the extension to `self.valid_extensions` in `PhotoScanner.__init__`.

### Switch CLIP Model

Change the `model_name` default in `MobileCLIPHandler.__init__`. Any `sentence-transformers` CLIP model works. Larger models (e.g. `clip-ViT-L-14`) give better accuracy at higher memory cost. **Note:** changing the model invalidates the existing FAISS index and all stored embeddings — delete `photos.db` and rescan.

### Switch to GPU for InsightFace

In `src/face_detector.py`, change:
```python
providers=["CPUExecutionProvider"]
```
to:
```python
providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
```

### Add a REST API

The module separation is clean enough that you can wrap `main.py`'s logic in a FastAPI server. The core components (`PhotoDatabase`, `FAISSIndex`, `MobileCLIPHandler`) are stateless enough to be instantiated once at startup and reused across requests.
