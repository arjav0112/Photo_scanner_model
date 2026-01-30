
import os
import time
from tqdm import tqdm
from .database import PhotoDatabase
from .model_handler import MobileCLIPHandler
import mimetypes
import numpy as np

class PhotoScanner:
    def __init__(self, model_path: str, db_path: str = "photos.db"):
        self.db = PhotoDatabase(db_path)
        print("Loading AI Model (this may take a moment)...")
        # Initialize robust SentenceTransformer model
        self.model = MobileCLIPHandler()
        
        # Initialize OCR (lazy load later or now)
        from .ocr_handler import OCRHandler
        self.ocr = OCRHandler()
        
        # Pre-compute document anchor embedding for Gatekeeper
        print("Initializing Gatekeeper (Document Detection)...")
        doc_text = "text document invoice receipt book page letter contract"
        self.doc_embedding = self.model.get_text_embedding(doc_text)
        self.doc_embedding /= np.linalg.norm(self.doc_embedding) # Normalize
        self.doc_threshold = 0.17 # Lowered from 0.21 based on experimentation

        # Supported extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def scan_directory(self, directory: str):
        """Recursively scans a directory for photos."""
        print(f"Scanning directory: {directory}")
        
        # Get set of already scanned paths
        scanned_paths = self.db.get_scanned_paths()
        
        # Collect new files
        new_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    full_path = os.path.abspath(os.path.join(root, file))
                    if full_path not in scanned_paths:
                        new_files.append(full_path)
        
        if not new_files:
            print("No new photos found.")
            return

        print(f"Found {len(new_files)} new photos. Processing in batches...")
        
        batch_size = 16 # Adjust based on RAM
        
        # Generator for batches
        def chunked(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        for batch_idx, batch_paths in enumerate(tqdm(chunked(new_files, batch_size), total=len(new_files)//batch_size + 1, desc="Scanning Batches")):
            try:
                if batch_idx % 5 == 0: # Log every 5 batches
                     self.model.log_memory(f"Scanning Batch {batch_idx}")
                     
                # 1. Batch Embeddings (The Slow Part Optimized)
                embeddings = self.model.get_image_embeddings_batch(batch_paths)
                
                if embeddings is None:
                    continue
                
                # Process each result in batch
                for i, path in enumerate(batch_paths):
                    embedding = embeddings[i]
                    # Skip flat zero embeddings (failed loads)
                    if np.all(embedding == 0):
                        continue
                        
                    # Normalize
                    emb_norm = embedding / np.linalg.norm(embedding)
                    
                    # 2. Gatekeeper
                    sim_score = np.dot(emb_norm, self.doc_embedding)
                    
                    ocr_text = ""
                    if sim_score > self.doc_threshold:
                         # Keep visual feedback clean
                         # tqdm.write(f"[DOC] {os.path.basename(path)}") 
                         ocr_text = self.ocr.extract_text(path)
                    
                    # 3. Metadata
                    metadata = self._extract_metadata(path)
                    
                    # 4. Save
                    try:
                        stat = os.stat(path)
                        self.db.add_photo(path, stat.st_size, stat.st_mtime, embedding, metadata, ocr_text)
                    except Exception as e:
                        print(f"DB Error {path}: {e}")
                        
            except Exception as e:
                print(f"Batch Error: {e}")

    def _extract_metadata(self, image_path: str) -> dict:
        """Extracts basic metadata (dimensions, EXIF) from image."""
        meta = {}
        try:
            from PIL import Image, ExifTags
            with Image.open(image_path) as img:
                meta['width'], meta['height'] = img.size
                meta['format'] = img.format
                
                # Basic EXIF
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        if tag_name == 'DateTimeOriginal':
                            meta['date_taken'] = str(value)
                        elif tag_name == 'Model':
                            meta['camera_model'] = str(value)
                        elif tag_name == 'Make':
                            meta['camera_make'] = str(value)
        except Exception:
            pass
        return meta

if __name__ == "__main__":
    # Test
    # Provide a directory path to scan
    import sys
    if len(sys.argv) > 1:
        scan_dir = sys.argv[1]
    else:
        scan_dir = "." # Scan current dir by default
    
    scanner = PhotoScanner(os.path.join("assets", "mobileclip_s1_datacompdr_first.tflite"))
    scanner.scan_directory(scan_dir)
