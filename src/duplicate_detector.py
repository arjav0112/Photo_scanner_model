
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import struct


# ── Perceptual Hash (pHash) ──────────────────────────────────────────────────

def compute_phash(image_path: str, hash_size: int = 16) -> Optional[int]:
    """
    Compute a perceptual hash (pHash) for an image using DCT.
    Returns a 256-bit integer (hash_size=16 → 16*16 = 256 bits), or None on failure.

    pHash is robust to:
      - minor brightness/contrast changes
      - slight resizing / compression artefacts
      - JPEG re-saves

    NOT robust to: heavy crops, rotations > ~5°, colour filters.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")                       # greyscale
            img = img.resize((hash_size, hash_size), Image.LANCZOS)
            pixels = np.array(img, dtype=np.float32)

            # 2-D DCT (hand-rolled; avoids scipy dependency)
            dct = _dct2(pixels)

            # Use top-left (hash_size x hash_size) low-frequency components
            # Exclude DC component (dct[0,0]) as it carries overall brightness
            dct_low = dct[:hash_size, :hash_size].flatten()
            dct_low = dct_low[1:]   # drop DC

            median = np.median(dct_low)
            bits = dct_low > median          # True → 1, False → 0

            # Pack into a Python integer
            hash_int = 0
            for b in bits:
                hash_int = (hash_int << 1) | int(b)
            return hash_int

    except Exception:
        return None


def _dct2(matrix: np.ndarray) -> np.ndarray:
    """Fast approximation of 2-D DCT-II using separable 1-D transforms."""
    return _dct1d(_dct1d(matrix, axis=0), axis=1)


def _dct1d(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """1-D DCT-II via numpy (no scipy needed)."""
    n = x.shape[axis]
    # Extend signal: [x, x_reversed] then take real RFFT
    if axis == 0:
        v = np.concatenate([x, x[::-1, :]], axis=0)
    else:
        v = np.concatenate([x, x[:, ::-1]], axis=1)
    V = np.fft.rfft(v, axis=axis)
    k = np.arange(n, dtype=np.float64)
    factor = 2 * np.exp(-1j * np.pi * k / (2 * n))
    if axis == 0:
        factor = factor[:, np.newaxis]
    result = np.real(V[:n] if axis == 0 else V[:, :n]) * np.real(factor[:n] if axis == 0 else factor[:, :n])
    return result


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Bit-count of XOR between two integer hashes."""
    return bin(hash_a ^ hash_b).count('1')


def phash_to_bytes(hash_int: int) -> bytes:
    """Serialise a pHash integer to bytes for SQLite storage."""
    return hash_int.to_bytes(32, byteorder='big')


def bytes_to_phash(blob: bytes) -> int:
    """Deserialise a pHash from bytes."""
    return int.from_bytes(blob, byteorder='big')


# ── Duplicate Group Detection ────────────────────────────────────────────────

class DuplicateDetector:
    """
    Finds duplicate and near-duplicate images using two complementary methods:

    1. Perceptual Hash (pHash)  — exact & near-exact pixel-level matches
       • Hamming distance ≤ PHASH_THRESHOLD → near-exact duplicate
       • Very fast: O(n²) bitmap comparisons, but we use early exit.

    2. CLIP Embedding Cosine Similarity — visually / semantically similar
       • Cosine similarity ≥ EMBEDDING_THRESHOLD → semantic near-duplicate
       • Slower, but catches edits, filters, crops that fool pHash.
    """

    # Hamming distance thresholds (lower = stricter)
    # 0  → bit-identical hash (pixel-perfect or near-identical JPEG)
    # ≤10 → strong near-duplicate (slight crop/resize/compression)
    # ≤20 → moderate near-duplicate (may include similar but distinct images)
    PHASH_EXACT_THRESHOLD   = 0    # pixel-identical
    PHASH_NEAR_THRESHOLD    = 10   # near-exact (default)

    # Embedding cosine similarity thresholds
    # 0.98 → essentially the same image rendered differently
    # 0.95 → very similar (same scene, slightly different framing)
    # 0.90 → semantically similar but possibly different shots
    EMBEDDING_HIGH_THRESHOLD = 0.98  # near-exact
    EMBEDDING_NEAR_THRESHOLD = 0.95  # near-duplicate (default)

    def __init__(self, phash_threshold: int = None, embedding_threshold: float = None):
        self.phash_threshold     = phash_threshold     if phash_threshold     is not None else self.PHASH_NEAR_THRESHOLD
        self.embedding_threshold = embedding_threshold if embedding_threshold is not None else self.EMBEDDING_NEAR_THRESHOLD

    # ── pHash-based grouping ─────────────────────────────────────────────────

    def find_phash_duplicates(
        self,
        entries: List[Dict]   # each: {'path': str, 'phash': int}
    ) -> List[List[str]]:
        """
        Groups images by pHash similarity (Hamming distance ≤ threshold).
        Returns list of duplicate groups (each group has ≥ 2 paths).
        Only paths whose phash != None are considered.
        """
        valid = [(e['path'], e['phash']) for e in entries if e.get('phash') is not None]
        n = len(valid)
        visited = [False] * n
        groups = []

        for i in range(n):
            if visited[i]:
                continue
            group = [valid[i][0]]
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                dist = hamming_distance(valid[i][1], valid[j][1])
                if dist <= self.phash_threshold:
                    group.append(valid[j][0])
                    visited[j] = True
            if len(group) > 1:
                visited[i] = True
                groups.append(group)

        return groups

    # ── Embedding-based grouping ─────────────────────────────────────────────

    def find_embedding_duplicates(
        self,
        paths: List[str],
        embeddings: np.ndarray   # shape (N, D), already L2-normalised
    ) -> List[List[str]]:
        """
        Groups images by CLIP embedding cosine similarity ≥ threshold.
        Returns list of duplicate groups (each group has ≥ 2 paths).
        Embeddings must be L2-normalised (dot product = cosine similarity).
        """
        n = len(paths)
        if n == 0:
            return []

        # Normalise just in case
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        emb_norm = embeddings / norms

        visited = [False] * n
        groups = []

        # Batch cosine similarity via matrix multiply
        # For large collections we chunk to avoid huge memory usage
        chunk_size = 512
        for i in range(n):
            if visited[i]:
                continue
            group = [paths[i]]
            # Compute similarities of i against i+1..n-1 in chunks
            for start in range(i + 1, n, chunk_size):
                end = min(start + chunk_size, n)
                sims = emb_norm[i] @ emb_norm[start:end].T
                for offset, sim in enumerate(sims):
                    j = start + offset
                    if not visited[j] and sim >= self.embedding_threshold:
                        group.append(paths[j])
                        visited[j] = True
            if len(group) > 1:
                visited[i] = True
                groups.append(group)

        return groups

    # ── Combined grouping ────────────────────────────────────────────────────

    def find_all_duplicates(
        self,
        entries: List[Dict],        # each: {'path', 'phash', 'embedding'}
        use_embedding: bool = True
    ) -> List[Dict]:
        """
        Master method combining pHash + embedding detection.

        Returns a list of duplicate-group dicts:
        {
            'type': 'exact' | 'near_exact' | 'semantic',
            'paths': [path1, path2, ...],   # all duplicates in this group
            'keep': path1,                   # suggested file to keep (first = largest)
            'remove': [path2, ...]           # suggested files to remove
        }
        """
        # ── Phase 1: pHash grouping ──
        phash_groups = self.find_phash_duplicates(entries)

        # Flatten phash-grouped paths into a set (they are handled)
        phash_covered = set()
        for g in phash_groups:
            phash_covered.update(g)

        # ── Phase 2: Embedding grouping (on remaining paths) ──
        emb_groups = []
        if use_embedding:
            remaining = [e for e in entries if e['path'] not in phash_covered and e.get('embedding') is not None]
            if remaining:
                paths_r   = [e['path'] for e in remaining]
                embs_r    = np.vstack([e['embedding'] for e in remaining])
                emb_groups = self.find_embedding_duplicates(paths_r, embs_r)

        # ── Build result groups ──
        result = []

        for group in phash_groups:
            # Classify by tightest hamming distance in group
            min_dist = self._min_hamming_in_group(group, entries)
            dup_type = 'exact' if min_dist <= self.PHASH_EXACT_THRESHOLD else 'near_exact'
            keep, remove = self._pick_representative(group, entries)
            result.append({
                'type':   dup_type,
                'paths':  group,
                'keep':   keep,
                'remove': remove
            })

        for group in emb_groups:
            keep, remove = self._pick_representative(group, entries)
            result.append({
                'type':   'semantic',
                'paths':  group,
                'keep':   keep,
                'remove': remove
            })

        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _min_hamming_in_group(self, group: List[str], entries: List[Dict]) -> int:
        path_to_phash = {e['path']: e.get('phash') for e in entries}
        min_dist = 999
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ha = path_to_phash.get(group[i])
                hb = path_to_phash.get(group[j])
                if ha is not None and hb is not None:
                    min_dist = min(min_dist, hamming_distance(ha, hb))
        return min_dist if min_dist < 999 else 0

    def _pick_representative(self, group: List[str], entries: List[Dict]) -> Tuple[str, List[str]]:
        """
        Choose which image to KEEP from a duplicate group.
        Strategy: keep the one with largest file size (typically highest quality).
        Ties broken by first appearance (original, not copy).
        """
        import os
        def file_size(p):
            try:
                return os.path.getsize(p)
            except OSError:
                return 0

        ranked = sorted(group, key=file_size, reverse=True)
        keep   = ranked[0]
        remove = ranked[1:]
        return keep, remove
