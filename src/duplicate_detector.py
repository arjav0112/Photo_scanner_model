
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional


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
    """
    2-D DCT-II via two separable 1-D passes (rows then columns).
    No scipy needed; works on any (H, W) float32 array.
    """
    return _dct1d_cols(_dct1d_rows(matrix))


def _dct1d_rows(x: np.ndarray) -> np.ndarray:
    """Apply 1-D DCT-II along axis=0 (each column independently)."""
    n = x.shape[0]
    # Mirror: [x ; flip(x,0)]  →  length-2n signal, then RFFT
    v = np.concatenate([x, x[::-1, :]], axis=0)          # (2n, W)
    V = np.fft.rfft(v, axis=0)                            # (n+1, W) complex
    k = np.arange(n, dtype=np.float64).reshape(-1, 1)    # (n, 1)
    factor = 2.0 * np.exp(-1j * np.pi * k / (2 * n))    # (n, 1)
    return np.real(V[:n, :]) * np.real(factor)            # (n, W)


def _dct1d_cols(x: np.ndarray) -> np.ndarray:
    """Apply 1-D DCT-II along axis=1 (each row independently)."""
    n = x.shape[1]
    v = np.concatenate([x, x[:, ::-1]], axis=1)          # (H, 2n)
    V = np.fft.rfft(v, axis=1)                            # (H, n+1) complex
    k = np.arange(n, dtype=np.float64).reshape(1, -1)    # (1, n)
    factor = 2.0 * np.exp(-1j * np.pi * k / (2 * n))    # (1, n)
    return np.real(V[:, :n]) * np.real(factor)            # (H, n)


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Bit-count of XOR between two integer hashes."""
    return bin(hash_a ^ hash_b).count('1')


def phash_to_bytes(hash_int: int) -> bytes:
    """Serialise a pHash integer to bytes for SQLite storage."""
    return hash_int.to_bytes(32, byteorder='big')


def bytes_to_phash(blob: bytes) -> int:
    """Deserialise a pHash from bytes."""
    return int.from_bytes(blob, byteorder='big')


# ── Union-Find (Disjoint Set) ────────────────────────────────────────────────

class UnionFind:
    """
    Path-compressed, rank-unioned disjoint set for O(α(n)) operations.
    Works on integer indices; use an external index↔path mapping.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def groups(self) -> Dict[int, List[int]]:
        """Return {root: [indices_in_group]} for groups with ≥ 2 members."""
        from collections import defaultdict
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            buckets[self.find(i)].append(i)
        return {r: members for r, members in buckets.items() if len(members) >= 2}


# ── BK-Tree for pHash ────────────────────────────────────────────────────────

class BKTree:
    """
    Burkhard-Keller tree over integer pHashes using Hamming distance.
    Allows sub-linear (O(log n) average) range queries for near-duplicates.
    """
    def __init__(self):
        # Each node: (hash_int, index, {dist: child_node})
        self._root: Optional[Tuple] = None

    def _make_node(self, h: int, idx: int) -> list:
        return [h, idx, {}]

    def insert(self, h: int, idx: int):
        if self._root is None:
            self._root = self._make_node(h, idx)
            return
        node = self._root
        while True:
            d = hamming_distance(h, node[0])
            if d == 0:
                return          # exact duplicate hash already in tree
            children = node[2]
            if d not in children:
                children[d] = self._make_node(h, idx)
                return
            node = children[d]

    def find_within(self, h: int, threshold: int) -> List[int]:
        """Return indices of all entries within Hamming distance ≤ threshold."""
        if self._root is None:
            return []
        results = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = hamming_distance(h, node[0])
            if d <= threshold:
                results.append(node[1])
            low  = max(1, d - threshold)
            high = d + threshold
            for dist, child in node[2].items():
                if low <= dist <= high:
                    stack.append(child)
        return results


# ── Duplicate Group Detection ────────────────────────────────────────────────

class DuplicateDetector:
    """
    Finds duplicate and near-duplicate images using two complementary methods:

    1. Perceptual Hash (pHash)  — exact & near-exact pixel-level matches
       • BK-tree range query:  O(n log n) average  (was O(n²))
       • Union-Find grouping:  O(n · α(n)) ≈ O(n)

    2. CLIP Embedding Cosine Similarity — visually / semantically similar
       • FAISS IVF or Flat index:  O(n log n) build + O(n · k) queries
       • Union-Find grouping

    Thresholds
    ----------
    pHash Hamming distance:
      0     → pixel-identical
      ≤ 10  → near-exact (slight crop/resize/compression)   [default]
      ≤ 20  → moderate near-duplicate

    Embedding cosine similarity:
      ≥ 0.98 → near-exact rendered differently
      ≥ 0.95 → near-duplicate (same scene, minor framing)   [default]
      ≥ 0.90 → semantically similar
    """

    PHASH_EXACT_THRESHOLD    = 0
    PHASH_NEAR_THRESHOLD     = 10

    EMBEDDING_HIGH_THRESHOLD = 0.98
    EMBEDDING_NEAR_THRESHOLD = 0.90

    def __init__(self, phash_threshold: int = None, embedding_threshold: float = None):
        self.phash_threshold     = phash_threshold     if phash_threshold     is not None else self.PHASH_NEAR_THRESHOLD
        self.embedding_threshold = embedding_threshold if embedding_threshold is not None else self.EMBEDDING_NEAR_THRESHOLD

    # ── pHash grouping  O(n log n) ───────────────────────────────────────────

    def find_phash_duplicates(
        self,
        entries: List[Dict]   # each: {'path': str, 'phash': int}
    ) -> List[List[str]]:
        """
        Groups images by pHash similarity using a BK-tree + Union-Find.
        Returns list of duplicate groups (each group has ≥ 2 paths).

        Complexity: O(n log n) average, vs O(n²) previously.
        """
        valid = [(e['path'], e['phash']) for e in entries if e.get('phash') is not None]
        n = len(valid)
        if n < 2:
            return []

        # Build BK-tree
        tree = BKTree()
        for idx, (_, h) in enumerate(valid):
            tree.insert(h, idx)

        # Union-Find over indices
        uf = UnionFind(n)

        for i, (_, h) in enumerate(valid):
            neighbours = tree.find_within(h, self.phash_threshold)
            for j in neighbours:
                if j != i:
                    uf.union(i, j)

        # Collect groups
        groups = []
        for members in uf.groups().values():
            paths = [valid[m][0] for m in members]
            groups.append(paths)

        return groups

    # ── Embedding grouping  O(n log n) ──────────────────────────────────────

    def find_embedding_duplicates(
        self,
        paths: List[str],
        embeddings: np.ndarray   # shape (N, D), already L2-normalised
    ) -> List[List[str]]:
        """
        Groups images by CLIP embedding cosine similarity ≥ threshold
        using FAISS for ANN search + Union-Find for grouping.

        Complexity: O(n log n) build + O(n · k) queries  (k = neighbours per image).
        Falls back to brute-force if FAISS is unavailable.
        """
        n = len(paths)
        if n < 2:
            return []

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        emb_norm = (embeddings / norms).astype(np.float32)

        uf = UnionFind(n)

        try:
            import faiss
            self._faiss_group(emb_norm, uf)
        except ImportError:
            self._bruteforce_group(emb_norm, uf)

        groups = []
        for members in uf.groups().values():
            paths_in_group = [paths[m] for m in members]
            groups.append(paths_in_group)

        return groups

    def _faiss_group(self, emb_norm: np.ndarray, uf: UnionFind):
        """FAISS inner-product search (= cosine similarity on normalised vecs)."""
        import faiss
        n, d = emb_norm.shape

        # Use IVF for large collections, Flat for small ones
        if n >= 1000:
            nlist = min(int(n ** 0.5), 256)
            quantiser = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(emb_norm)
            index.nprobe = max(1, nlist // 4)
        else:
            index = faiss.IndexFlatIP(d)

        index.add(emb_norm)

        # For each image retrieve its k nearest neighbours
        k = min(16, n)
        sims, indices = index.search(emb_norm, k)   # (n, k)

        threshold = float(self.embedding_threshold)
        for i in range(n):
            for rank in range(k):
                j   = int(indices[i, rank])
                sim = float(sims[i, rank])
                if j == i or j < 0:
                    continue
                if sim >= threshold:
                    uf.union(i, j)
                else:
                    break   # FAISS results are sorted by similarity desc

    def _bruteforce_group(self, emb_norm: np.ndarray, uf: UnionFind):
        """Fallback O(n²) cosine grouping when FAISS is not installed."""
        n = len(emb_norm)
        chunk = 512
        threshold = float(self.embedding_threshold)
        for i in range(n):
            for start in range(i + 1, n, chunk):
                end = min(start + chunk, n)
                sims = emb_norm[i] @ emb_norm[start:end].T
                for offset, sim in enumerate(sims):
                    if sim >= threshold:
                        uf.union(i, start + offset)

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
            'paths': [path1, path2, ...],
            'keep': path1,                   # largest file (highest quality)
            'remove': [path2, ...]
        }
        """
        # ── Phase 1: pHash grouping ──
        phash_groups = self.find_phash_duplicates(entries)

        phash_covered = set()
        for g in phash_groups:
            phash_covered.update(g)

        # ── Phase 2: Embedding grouping (on remaining paths only) ──
        emb_groups = []
        if use_embedding:
            remaining = [e for e in entries if e['path'] not in phash_covered and e.get('embedding') is not None]
            if len(remaining) >= 2:
                paths_r = [e['path'] for e in remaining]
                embs_r  = np.vstack([e['embedding'] for e in remaining])
                emb_groups = self.find_embedding_duplicates(paths_r, embs_r)

        # ── Build result dicts ──
        result = []

        path_to_phash = {e['path']: e.get('phash') for e in entries}

        for group in phash_groups:
            min_dist = self._min_hamming_in_group(group, path_to_phash)
            dup_type = 'exact' if min_dist <= self.PHASH_EXACT_THRESHOLD else 'near_exact'
            keep, remove = self._pick_representative(group)
            result.append({'type': dup_type, 'paths': group, 'keep': keep, 'remove': remove})

        for group in emb_groups:
            keep, remove = self._pick_representative(group)
            result.append({'type': 'semantic', 'paths': group, 'keep': keep, 'remove': remove})

        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _min_hamming_in_group(self, group: List[str], path_to_phash: Dict) -> int:
        min_dist = 999
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ha = path_to_phash.get(group[i])
                hb = path_to_phash.get(group[j])
                if ha is not None and hb is not None:
                    min_dist = min(min_dist, hamming_distance(ha, hb))
        return min_dist if min_dist < 999 else 0

    def _pick_representative(self, group: List[str]) -> Tuple[str, List[str]]:
        """
        Keep the largest file (typically highest quality).
        Ties broken by lexicographic order (stable, reproducible).
        """
        import os
        def file_size(p):
            try:
                return os.path.getsize(p)
            except OSError:
                return 0

        ranked = sorted(group, key=lambda p: (file_size(p), p), reverse=True)
        return ranked[0], ranked[1:]
