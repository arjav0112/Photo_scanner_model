
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
        if "phash" not in columns:
            print("Migrating Database: Adding 'phash' column...")
            cursor.execute("ALTER TABLE photos ADD COLUMN phash BLOB")
        if "is_duplicate" not in columns:
            print("Migrating Database: Adding 'is_duplicate' column...")
            cursor.execute("ALTER TABLE photos ADD COLUMN is_duplicate INTEGER DEFAULT 0")
        if "duplicate_of" not in columns:
            print("Migrating Database: Adding 'duplicate_of' column...")
            cursor.execute("ALTER TABLE photos ADD COLUMN duplicate_of TEXT")

        # --- persons table -----------------------------------------------
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            representative_face_id INTEGER,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            updated_at REAL DEFAULT (strftime('%s', 'now'))
        )
        ''')

        # --- faces table -------------------------------------------------
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            person_id INTEGER REFERENCES persons(id),
            bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
            det_score REAL,
            embedding BLOB NOT NULL,
            age_estimate INTEGER,
            gender TEXT
        )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)')

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

    def update_phash(self, path: str, phash_bytes: bytes):
        """Store the perceptual hash blob for a photo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE photos SET phash=? WHERE path=?", (phash_bytes, path))
        conn.commit()
        conn.close()

    def get_all_for_dedup(self) -> List[dict]:
        """
        Retrieve all photos with id, path, size_bytes, phash, and embedding
        for duplicate detection. Used by the 'dedupe' command.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, path, size_bytes, phash, embedding FROM photos "
            "WHERE embedding IS NOT NULL ORDER BY id"
        )
        rows = cursor.fetchall()
        conn.close()

        result = []
        for row_id, path, size, phash_blob, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None
            phash_int = int.from_bytes(phash_blob, byteorder='big') if phash_blob else None
            result.append({
                'id':         row_id,
                'path':       path,
                'size_bytes': size,
                'phash':      phash_int,
                'embedding':  emb,
            })
        return result

    def mark_as_duplicate(self, path: str, original_path: str):
        """Flag a photo as a duplicate of another."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE photos SET is_duplicate=1, duplicate_of=? WHERE path=?",
            (original_path, path)
        )
        conn.commit()
        conn.close()

    def unmark_duplicate(self, path: str):
        """Remove the duplicate flag from a photo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE photos SET is_duplicate=0, duplicate_of=NULL WHERE path=?",
            (path,)
        )
        conn.commit()
        conn.close()

    def get_duplicates(self) -> List[dict]:
        """Return all photos currently flagged as duplicates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT path, duplicate_of FROM photos WHERE is_duplicate=1 ORDER BY path"
        )
        rows = cursor.fetchall()
        conn.close()
        return [{'path': r[0], 'duplicate_of': r[1]} for r in rows]

    def delete_photos_by_path(self, paths: List[str]):
        """Permanently delete photo records from the database."""
        if not paths:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(paths))
        cursor.execute(f"DELETE FROM photos WHERE path IN ({placeholders})", paths)
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        print(f"Removed {deleted} duplicate record(s) from database.")

    # =========================================================================
    # Face / Person CRUD
    # =========================================================================

    def get_photo_id(self, path: str) -> Optional[int]:
        """Return the integer PK for a photo path, or None."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE path=?", (path,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def add_face(self, photo_id: int, bbox: list, det_score: float,
                 embedding: "np.ndarray", age: Optional[int] = None,
                 gender: Optional[str] = None) -> int:
        """Insert a face record; returns the new face id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        emb_blob = embedding.astype(np.float32).tobytes()
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cursor.execute(
            '''INSERT INTO faces
               (photo_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                det_score, embedding, age_estimate, gender)
               VALUES (?,?,?,?,?,?,?,?,?)''',
            (photo_id, x1, y1, x2, y2, det_score, emb_blob, age, gender),
        )
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return face_id

    def delete_faces_for_photo(self, photo_id: int):
        """Remove all face rows for a given photo (used on rescan)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
        conn.commit()
        conn.close()

    def get_all_faces_with_embeddings(self) -> List[dict]:
        """Return all face rows with embeddings for clustering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, photo_id, person_id, embedding FROM faces WHERE embedding IS NOT NULL"
        )
        rows = cursor.fetchall()
        conn.close()
        result = []
        for face_id, photo_id, person_id, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
            result.append({"id": face_id, "photo_id": photo_id, "person_id": person_id, "embedding": emb})
        return result

    def update_face_person(self, face_id: int, person_id: Optional[int]):
        """Assign a person_id to a face row."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
        conn.commit()
        conn.close()

    def add_person(self, name: Optional[str] = None, rep_face_id: Optional[int] = None) -> int:
        """Insert a new person; returns new person id."""
        import time as _time
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = _time.time()
        cursor.execute(
            "INSERT INTO persons (name, representative_face_id, created_at, updated_at) VALUES (?,?,?,?)",
            (name, rep_face_id, now, now),
        )
        pid = cursor.lastrowid
        conn.commit()
        conn.close()
        return pid

    def update_person_name(self, person_id: int, name: str):
        """Assign a human-readable name to a person cluster."""
        import time as _time
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE persons SET name=?, updated_at=? WHERE id=?",
                       (name, _time.time(), person_id))
        conn.commit()
        conn.close()

    def get_persons(self) -> List[dict]:
        """Return all persons with face counts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT p.id, p.name, p.representative_face_id, COUNT(f.id) as face_count
               FROM persons p
               LEFT JOIN faces f ON f.person_id = p.id
               GROUP BY p.id ORDER BY face_count DESC'''
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "representative_face_id": r[2], "face_count": r[3]}
                for r in rows]

    def get_photos_for_person(self, person_id: int) -> List[str]:
        """Return distinct photo paths containing a given person."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT DISTINCT p.path FROM faces f
               JOIN photos p ON p.id = f.photo_id
               WHERE f.person_id = ? ORDER BY p.path''',
            (person_id,),
        )
        paths = [r[0] for r in cursor.fetchall()]
        conn.close()
        return paths

    def get_faces_for_photo(self, photo_path: str) -> List[dict]:
        """Return all face metadata for a photo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT f.id, f.person_id, f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
                      f.det_score, f.age_estimate, f.gender, pe.name
               FROM faces f
               JOIN photos ph ON ph.id = f.photo_id
               LEFT JOIN persons pe ON pe.id = f.person_id
               WHERE ph.path = ?''',
            (photo_path,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"face_id": r[0], "person_id": r[1], "bbox": [r[2], r[3], r[4], r[5]],
                 "det_score": r[6], "age": r[7], "gender": r[8], "person_name": r[9]}
                for r in rows]

    def get_person_centroids(self) -> dict:
        """Return {person_id: centroid_embedding} for incremental assignment."""
        faces = self.get_all_faces_with_embeddings()
        by_person: dict = {}
        for f in faces:
            pid = f["person_id"]
            if pid is None:
                continue
            by_person.setdefault(pid, []).append(f["embedding"])
        centroids = {}
        for pid, embs in by_person.items():
            stack = np.vstack(embs)
            c = stack.mean(axis=0)
            norm = np.linalg.norm(c)
            if norm > 1e-6:
                c /= norm
            centroids[pid] = c.astype(np.float32)
        return centroids

    def delete_all_persons(self):
        """Wipe persons table and person_id assignments (for full recluster)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE faces SET person_id=NULL")
        cursor.execute("DELETE FROM persons")
        conn.commit()
        conn.close()




if __name__ == "__main__":
    db = PhotoDatabase()
    print(f"Database initialized at {db.db_path}")
