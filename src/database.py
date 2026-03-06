
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
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_path ON photos(path)')
        
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
        embedding_blob = embedding.astype(np.float32).tobytes()
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        try:
            cursor.execute('''
            INSERT INTO photos (path, filename, size_bytes, modified_time, embedding, metadata, ocr_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (path, filename, size, mtime, embedding_blob, metadata_json, ocr_text))
            conn.commit()
        except sqlite3.IntegrityError:
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
        
        cursor.execute("PRAGMA table_info(photos)")
        cols = [c[1] for c in cursor.fetchall()]
        has_ocr = "ocr_text" in cols
        
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
                
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                metadata_list.append(metadata)
            
            yield paths, np.vstack(embeddings).astype(np.float32), ocr_texts, metadata_list
            
        conn.close()
    
    def get_all_embeddings_with_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve all (id, embedding) pairs for FAISS index building.
        
        Returns:
            (ids, embeddings): numpy arrays
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM photos WHERE embedding IS NOT NULL")
        
        ids = []
        embeddings = []
        
        for row in cursor.fetchall():
            row_id, blob = row
            emb = np.frombuffer(blob, dtype=np.float32)
            ids.append(row_id)
            embeddings.append(emb)
        
        conn.close()
        
        if not embeddings:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        
        return np.array(ids, dtype=np.int64), np.vstack(embeddings).astype(np.float32)
    
    def get_batch_by_ids(self, ids: list) -> list:
        """
        Retrieve photo data for specific database IDs (post-FAISS search).
        
        Args:
            ids: List of database row IDs
            
        Returns:
            List of dicts with path, ocr_text, metadata for each ID
        """
        if len(ids) == 0:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(ids))
        cursor.execute(f'''
        SELECT id, path, ocr_text, metadata 
        FROM photos 
        WHERE id IN ({placeholders})
        ''', list(ids))
        
        result_map = {}
        for row in cursor.fetchall():
            row_id, path, ocr_text, metadata_json = row
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            
            result_map[row_id] = {
                'id': row_id,
                'path': path,
                'ocr_text': ocr_text or '',
                'metadata': metadata
            }
        
        conn.close()
        
        return [result_map[i] for i in ids if i in result_map]
    
    def get_photo_count(self) -> int:
        """Get total number of photos with embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_scanned_paths_with_mtime(self) -> dict:
        """
        Returns dict of {path: (modified_time, size_bytes)} for change detection.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path, modified_time, size_bytes FROM photos")
        result = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        conn.close()
        return result
    
    def update_photo(self, path: str, size: int, mtime: float, embedding: np.ndarray, metadata: dict = None, ocr_text: str = ""):
        """Update an existing photo entry (for incremental re-scanning)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = embedding.astype(np.float32).tobytes()
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        cursor.execute('''
        UPDATE photos 
        SET size_bytes=?, modified_time=?, embedding=?, metadata=?, ocr_text=?
        WHERE path=?
        ''', (size, mtime, embedding_blob, metadata_json, ocr_text, path))
        
        conn.commit()
        conn.close()
    
    def remove_photos(self, paths: list):
        """Remove photos that no longer exist on disk."""
        if not paths:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(paths))
        cursor.execute(f"DELETE FROM photos WHERE path IN ({placeholders})", paths)
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        if deleted > 0:
            print(f"Removed {deleted} deleted files from database")

if __name__ == "__main__":
    db = PhotoDatabase()
    print(f"Database initialized at {db.db_path}")
