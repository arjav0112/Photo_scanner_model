
import sqlite3
import numpy as np
import io
import os
import json
from typing import List, Tuple, Optional

class PhotoDatabase:
    def __init__(self, db_path: str = "photos.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for photos
        # We store the embedding as a BLOB
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT,
            size_bytes INTEGER,
            modified_time REAL,
            embedding BLOB,
            metadata TEXT
        )
        ''')
        
        # Create index on path for fast lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_path ON photos(path)')
        
        # Schema Migration: Check if ocr_text exists
        cursor.execute("PRAGMA table_info(photos)")
        columns = [info[1] for info in cursor.fetchall()]
        if "ocr_text" not in columns:
            print("Migrating Database: Adding 'ocr_text' column...")
            cursor.execute("ALTER TABLE photos ADD COLUMN ocr_text TEXT")
        
        conn.commit()
        conn.close()

    def get_scanned_paths(self) -> set:
        """Returns a set of all file paths currently in the DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM photos")
        paths = set(row[0] for row in cursor.fetchall())
        conn.close()
        return paths

    def add_photo(self, path: str, size: int, mtime: float, embedding: np.ndarray, metadata: dict = None, ocr_text: str = ""):
        """Adds a photo and its embedding to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        filename = os.path.basename(path)
        # Convert numpy array to bytes
        embedding_blob = embedding.astype(np.float32).tobytes()
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        try:
            cursor.execute('''
            INSERT INTO photos (path, filename, size_bytes, modified_time, embedding, metadata, ocr_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (path, filename, size, mtime, embedding_blob, metadata_json, ocr_text))
            conn.commit()
        except sqlite3.IntegrityError:
            # Already exists, maybe update? For now, ignore or print.
            print(f"File already in DB: {path}")
        
        conn.close()

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves all embeddings and their corresponding paths.
        Useful for in-memory search or building an index.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path, embedding FROM photos WHERE embedding IS NOT NULL")
        
        paths = []
        embeddings = []
        
        for row in cursor.fetchall():
            path, blob = row
            # Convert bytes back to numpy array
            emb = np.frombuffer(blob, dtype=np.float32)
            paths.append(path)
            embeddings.append(emb)
            
        conn.close()
        
        if not embeddings:
            return [], np.array([], dtype=np.float32)
            
        return paths, np.vstack(embeddings).astype(np.float32)

    def get_all_search_data(self) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Retrieves paths, embeddings, and OCR text for search.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if ocr_text column exists first (backward compatibility)
        cursor.execute("PRAGMA table_info(photos)")
        cols = [c[1] for c in cursor.fetchall()]
        has_ocr = "ocr_text" in cols
        
        query = "SELECT path, embedding, " + ("ocr_text " if has_ocr else "'' ") + "FROM photos WHERE embedding IS NOT NULL"
        cursor.execute(query)
        
        paths = []
        embeddings = []
        ocr_texts = []
        
        for row in cursor.fetchall():
            path, blob, text = row
            emb = np.frombuffer(blob, dtype=np.float32)
            paths.append(path)
            embeddings.append(emb)
            ocr_texts.append(text if text else "")
            
        conn.close()
        
        if not embeddings:
            return [], np.array([], dtype=np.float32), []
            
        return paths, np.vstack(embeddings).astype(np.float32), ocr_texts

    def get_search_data_generator(self, batch_size: int = 1000):
        """
        Yields batches of (paths, embeddings, ocr_texts, metadata_list) for memory-efficient search.
        Now includes metadata for metadata-aware search.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check OCR column
        cursor.execute("PRAGMA table_info(photos)")
        cols = [c[1] for c in cursor.fetchall()]
        has_ocr = "ocr_text" in cols
        
        # Now also retrieve metadata
        query = "SELECT path, embedding, " + ("ocr_text, " if has_ocr else "'', ") + "metadata FROM photos WHERE embedding IS NOT NULL"
        cursor.execute(query)
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            paths = []
            embeddings = []
            ocr_texts = []
            metadata_list = []
            
            for row in rows:
                path, blob, text, metadata_json = row
                emb = np.frombuffer(blob, dtype=np.float32)
                paths.append(path)
                embeddings.append(emb)
                ocr_texts.append(text if text else "")
                
                # Parse metadata JSON
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                metadata_list.append(metadata)
            
            yield paths, np.vstack(embeddings).astype(np.float32), ocr_texts, metadata_list
            
        conn.close()

if __name__ == "__main__":
    db = PhotoDatabase()
    print(f"Database initialized at {db.db_path}")
