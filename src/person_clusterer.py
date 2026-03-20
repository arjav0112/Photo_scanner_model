
"""
Person Clustering: groups face embeddings into identity clusters.

Algorithm:
  1. Build a cosine-similarity graph of all face embeddings.
  2. Apply DBSCAN to find dense clusters.
  3. Merge chained clusters via connected-components to catch cross-age cases
     where A→B and B→C are similar but A→C is not (identity drift).
  4. Assign person_id labels back to individual face records.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
COSINE_THRESHOLD = 0.45   # distance below this = same person (cosine space)
MIN_CLUSTER_SIZE = 2       # faces needed to form a confirmed person group
CHAIN_THRESHOLD  = 0.50   # looser threshold for graph-merge chaining


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return pairwise cosine distance matrix (N x N)."""
    # embeddings are already L2-normalised so cosine_distance = 1 - dot product
    sims = embeddings @ embeddings.T
    sims = np.clip(sims, -1.0, 1.0)
    return 1.0 - sims


def _dbscan_cluster(dist_matrix: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Simple DBSCAN implementation that works on a precomputed distance matrix.
    Returns label array where -1 = noise.
    """
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    labels = db.fit_predict(dist_matrix)
    return labels


def _graph_merge(
    dist_matrix: np.ndarray,
    dbscan_labels: np.ndarray,
    chain_threshold: float,
) -> np.ndarray:
    """
    After DBSCAN, build an adjacency graph using a slightly looser threshold
    and merge clusters that are connected through intermediate nodes.
    This handles cross-age identity drift chains.

    Returns updated labels array.
    """
    n = len(dbscan_labels)

    # Build adjacency matrix for ALL faces (including noise)
    adj = (dist_matrix < chain_threshold).astype(np.uint8)
    np.fill_diagonal(adj, 0)

    sparse_adj = csr_matrix(adj)
    n_components, component_labels = connected_components(
        sparse_adj, directed=False, return_labels=True
    )

    # component_labels is now the merged identity label (0-indexed)
    # Noise points from DBSCAN that form isolated nodes get their own component
    # Small components (size < MIN_CLUSTER_SIZE) stay as unknowns (-1)
    component_sizes = np.bincount(component_labels, minlength=n_components)

    merged = np.full(n, -1, dtype=np.int32)
    person_id = 0
    for comp_idx in range(n_components):
        if component_sizes[comp_idx] >= MIN_CLUSTER_SIZE:
            mask = component_labels == comp_idx
            merged[mask] = person_id
            person_id += 1

    return merged


def cluster_faces(
    face_embeddings: np.ndarray,  # shape (N, 512)
    face_ids: List[int],          # DB ids of the face rows
) -> Dict[int, int]:
    """
    Cluster a list of face embeddings and return a mapping:
        { face_db_id → person_cluster_label }
    where person_cluster_label == -1 means unassigned.

    This is called on the FULL set of face embeddings (or a refreshed batch)
    so that cross-photo links can be formed.
    """
    if len(face_embeddings) == 0:
        return {}

    n = len(face_embeddings)

    if n == 1:
        # Single face — can't cluster, label as unknown
        return {face_ids[0]: -1}

    dist_matrix = _cosine_distance_matrix(face_embeddings)
    dist_matrix = dist_matrix.astype(np.float64)

    # Step 1: DBSCAN
    dbscan_labels = _dbscan_cluster(
        dist_matrix, eps=COSINE_THRESHOLD, min_samples=MIN_CLUSTER_SIZE
    )

    # Step 2: Graph-merge for cross-age chaining
    merged_labels = _graph_merge(dist_matrix, dbscan_labels, CHAIN_THRESHOLD)

    return {face_ids[i]: int(merged_labels[i]) for i in range(n)}


def compute_cluster_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute L2-normalised centroid of a set of embeddings."""
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-6:
        centroid /= norm
    return centroid.astype(np.float32)


def assign_new_face(
    new_embedding: np.ndarray,          # 512-d, already normalised
    person_centroids: Dict[int, np.ndarray],  # person_id → centroid embedding
    threshold: float = COSINE_THRESHOLD,
) -> Optional[int]:
    """
    Given a new face embedding and existing person centroids, return the
    best-matching person_id (cosine distance < threshold), or None if no match.
    Used for incremental assignment when new photos are scanned.
    """
    if not person_centroids:
        return None

    best_person = None
    best_dist = float("inf")

    for pid, centroid in person_centroids.items():
        dist = 1.0 - float(np.dot(new_embedding, centroid))
        if dist < best_dist:
            best_dist = dist
            best_person = pid

    if best_dist < threshold:
        return best_person
    return None
