"""
FAISS Vector Index for fast similarity search.
Uses IndexFlatIP (inner product) for exact cosine similarity.
"""

import faiss
import numpy as np
import os
import time
from typing import Tuple, Optional


class FAISSIndex:
    """Manages a FAISS index for fast vector similarity search."""
    
    INDEX_FILE = "faiss_index.bin"
    ID_MAP_FILE = "faiss_id_map.npy"
    
    def __init__(self, index_dir: str = "."):
        """
        Args:
            index_dir: Directory to store/load the FAISS index files
        """
        self.index_dir = index_dir
        self.index = None
        self.id_map = None
        self.dimension = None
    
    def build_index(self, ids: np.ndarray, embeddings: np.ndarray):
        """
        Build a new FAISS index from embeddings.
        
        Args:
            ids: Array of database row IDs
            embeddings: (N, D) float32 array of normalized embeddings
        """
        n, d = embeddings.shape
        self.dimension = d
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        self.id_map = ids.copy()
        
        print(f"FAISS index built: {n} vectors, {d} dimensions")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for most similar vectors.
        
        Args:
            query_embedding: (D,) query vector
            top_k: Number of results to return
            
        Returns:
            (db_ids, scores): Arrays of database IDs and similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        
        query = query_embedding.copy().reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)
        
        top_k = min(top_k, self.index.ntotal)
        
        scores, indices = self.index.search(query, top_k)
        
        valid = indices[0] >= 0
        faiss_indices = indices[0][valid]
        result_scores = scores[0][valid]
        db_ids = self.id_map[faiss_indices]
        
        return db_ids, result_scores
    
    def add_vectors(self, ids: np.ndarray, embeddings: np.ndarray):
        """
        Add new vectors to existing index (incremental).
        
        Args:
            ids: Database row IDs for new vectors
            embeddings: (N, D) float32 embeddings
        """
        if self.index is None:
            self.build_index(ids, embeddings)
            return
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        self.index.add(embeddings)
        self.id_map = np.concatenate([self.id_map, ids])
    
    def save(self):
        """Save index and ID map to disk."""
        if self.index is None:
            return
        
        index_path = os.path.join(self.index_dir, self.INDEX_FILE)
        map_path = os.path.join(self.index_dir, self.ID_MAP_FILE)
        
        faiss.write_index(self.index, index_path)
        np.save(map_path, self.id_map)
        print(f"FAISS index saved ({self.index.ntotal} vectors)")
    
    def load(self) -> bool:
        """
        Load index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.index_dir, self.INDEX_FILE)
        map_path = os.path.join(self.index_dir, self.ID_MAP_FILE)
        
        if not os.path.exists(index_path) or not os.path.exists(map_path):
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            self.id_map = np.load(map_path)
            self.dimension = self.index.d
            print(f"FAISS index loaded ({self.index.ntotal} vectors)")
            return True
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
            return False
    
    def load_or_build(self, db) -> None:
        """
        Load index from disk or build from database.
        
        Args:
            db: PhotoDatabase instance
        """
        if self.load():
            db_count = db.get_photo_count()
            if self.index.ntotal == db_count:
                return
            print(f"Index out of date ({self.index.ntotal} vs {db_count} in DB). Rebuilding...")
        
        start = time.time()
        ids, embeddings = db.get_all_embeddings_with_ids()
        
        if len(ids) == 0:
            print("No embeddings in database to index.")
            return
        
        self.build_index(ids, embeddings)
        self.save()
        elapsed = time.time() - start
        print(f"Index built and saved in {elapsed:.2f}s")
    
    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0
