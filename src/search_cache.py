"""
Search Cache with DISK-BASED embedding persistence.
Embeddings survive across process restarts — avoids model loading for known queries.
"""

import time
import hashlib
import numpy as np
import os
import json
from collections import OrderedDict
from typing import Optional, List, Dict


class SearchCache:
    """Hybrid cache: in-memory LRU for results, disk-based for embeddings."""
    
    EMBEDDING_DIR = "embedding_cache"
    
    def __init__(self, max_entries: int = 50, ttl_seconds: int = 300, cache_dir: str = "."):
        """
        Args:
            max_entries: Maximum in-memory cached queries
            ttl_seconds: TTL for in-memory results cache (default: 5 min)
            cache_dir: Directory for disk-based embedding cache
        """
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._cache = OrderedDict()  
        
        self._emb_dir = os.path.join(cache_dir, self.EMBEDDING_DIR)
        os.makedirs(self._emb_dir, exist_ok=True)
    
    def _make_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _evict_expired(self):
        """Remove expired in-memory entries."""
        now = time.time()
        expired = [k for k, (ts, _) in self._cache.items() if now - ts > self.ttl]
        for k in expired:
            del self._cache[k]
    
    def _evict_lru(self):
        """Evict least recently used if over capacity."""
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
    
    def get_text_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get cached text embedding from disk. No model loading needed."""
        key = self._make_key(query)
        emb_path = os.path.join(self._emb_dir, f"{key}.npy")
        
        if os.path.exists(emb_path):
            try:
                return np.load(emb_path)
            except Exception:
                pass
        return None
    
    def set_text_embedding(self, query: str, embedding: np.ndarray):
        """Save text embedding to disk for future sessions."""
        key = self._make_key(query)
        emb_path = os.path.join(self._emb_dir, f"{key}.npy")
        try:
            np.save(emb_path, embedding.astype(np.float32))
        except Exception:
            pass
    
    def get_results(self, query: str) -> Optional[List[Dict]]:
        """Get cached search results (in-memory only)."""
        self._evict_expired()
        key = self._make_key(query)
        
        if key in self._cache:
            ts, results = self._cache[key]
            self._cache.move_to_end(key)
            return results
        return None
    
    def set_results(self, query: str, results: List[Dict]):
        """Cache search results in memory."""
        key = self._make_key(query)
        self._cache[key] = (time.time(), results)
        self._evict_lru()
    
    def invalidate(self):
        """Clear in-memory cache (disk embeddings stay — they're query-independent)."""
        self._cache.clear()
    
    def invalidate_all(self):
        """Clear everything including disk cache."""
        self._cache.clear()
        for f in os.listdir(self._emb_dir):
            if f.endswith('.npy'):
                try:
                    os.remove(os.path.join(self._emb_dir, f))
                except Exception:
                    pass
    
    def stats(self) -> Dict:
        """Return cache statistics."""
        self._evict_expired()
        disk_count = len([f for f in os.listdir(self._emb_dir) if f.endswith('.npy')])
        return {
            'result_entries': len(self._cache),
            'disk_embeddings': disk_count,
            'max_entries': self.max_entries,
            'ttl_seconds': self.ttl
        }
