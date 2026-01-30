# Photo Scanner & AI Search

A powerful local photo organizer that uses AI to scan, analyze, and search your personal photo collection. It runs entirely offline using local Deep Learning models.

## Features

*   **Smart Scanning**: Recursively scans directories for images (`.jpg`, `.png`, `.webp`, etc.).
*   **AI-Powered Search**: Search your photos using natural language (e.g., "dog on the beach", "receipt for dinner"). Uses **CLIP (ViT-B-32)** for state-of-the-art text-to-image matching.
*   **Intelligent OCR**: Automatically detects documents (receipts, invoices, letters) using a "Gatekeeper" system and extracts text for keyword searching.
*   **Hybrid Search**: Combines visual understanding (Vector Search) with text extraction (OCR) for maximum accuracy.
*   **Privacy First**: All processing happens locally on your machine. Your photos never leave your computer.
*   **Mobile Ready**: Includes scripts to export processed data for mobile apps (quantized ONNX models).

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

1.  **Python 3.12+**
2.  **Poetry** installed globally.

### Installation

1.  Navigate to the project directory:
    ```bash
    cd Photo_scanner
    ```

2.  Install dependencies:
    ```bash
    poetry install
    ```

## Usage

The main entry point is `main.py`.

### 1. Scan Photos

This step analyzes your images and creates a local database (`photos.db`). It creates embeddings for every photo and extracts text from documents.

```bash
# Scan a specific directory
poetry run python main.py scan "C:\Users\YourName\Pictures"

# Or scan the current directory
poetry run python main.py scan .
```

*Note: The first run will download the CLIP model from HuggingFace if the local offline model is not found in `assets/local_clip_model`.*

### 2. Search Photos

Once scanned, you can search your library instantly.

```bash
poetry run python main.py search "birthday cake"
poetry run python main.py search "invoice January"
```

The results are displayed in the terminal and saved to `search_results.log`.

## Architecture & Models

### AI Engine
*   **Visual/Text Embedding**: Uses `sentence_transformers` with the `clip-ViT-B-32` architecture.
*   **OCR**: Standard OCR handler (Tesseract/EasyOCR wrapper usually, check `src/ocr_handler.py`).

### Local Offline Support
The system is designed to run offline.
*   If `assets/local_clip_model/` exists, the system loads the model from there.
*   Otherwise, it attempts to download `clip-ViT-B-32` from HuggingFace and cache it.

### Export to Mobile
The project contains experimental support for exporting the database and quantized models (`.onnx`) for use in a mobile flutter/android application.
*   `export_to_mobile.py`: Exports the SQLite database to a mobile-compatible format.
*   `assets/mobile_model_quantized/`: Contains the quantized ONNX versions of the models.

## Structure
*   `main.py`: CLI entry point.
*   `src/scanner.py`: Main scanning logic (Batch processing, Gatekeeper).
*   `src/model_handler.py`: Interface for the AI models (CLIP).
*   `src/database.py`: SQLite database queries.
*   `src/ocr_handler.py`: Text extraction logic.
