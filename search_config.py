"""
Search Configuration
Adjust these parameters to control search result filtering
"""

class SearchConfig:
    """
    Configuration for adaptive threshold filtering in search results.

    Threshold formula (3-way adaptive):
      relative_cut = top_score * (1 - RELATIVE_THRESHOLD)   # must be within X% of top
      floor_cut    = top_score * FLOOR_RATIO                 # never below Y% of top
      threshold    = max(relative_cut, floor_cut)            # take the tighter of the two

    This removes the hard MIN_ABSOLUTE_SCORE wall so that when top scores
    are naturally low (e.g. visual-only parrot queries score ~0.20-0.29),
    near-equivalent images are NOT blocked by a fixed absolute number.
    """
    
    # ========== MAIN PARAMETERS (Adjust these) ==========

    # Relative Threshold: how far below the top score is still acceptable.
    # threshold = top_score * (1 - RELATIVE_THRESHOLD)
    # 0.30 means images within 30% of the top score are included.
    # Range: 0.20 (tighter) → 0.40 (more lenient)
    RELATIVE_THRESHOLD = 0.30

    # Floor Ratio: hard lower bound expressed as a fraction of the top score.
    # threshold >= top_score * FLOOR_RATIO
    # 0.70 means the threshold is never below 70% of the top score,
    # keeping results tightly clustered even when top score is high.
    # Range: 0.60 (lenient) → 0.85 (strict)
    FLOOR_RATIO = 0.70

    # Maximum Results: Safety limit to prevent overwhelming output
    # Recommended range: 20 to 100
    MAX_RESULTS = 50
    
    # Metadata Weight: Controls how much metadata matching contributes to final score
    # NOTE: This is deprecated in favor of dynamic weights from query_analyzer
    # Kept for backward compatibility
    # Recommended range: 0.3 to 1.0
    METADATA_WEIGHT = 0.5
    
    # ========== DYNAMIC WEIGHT CONFIGURATION ==========
    
    # Weight presets for different query intents
    # These can be adjusted based on user feedback
    WEIGHT_PRESETS = {
        'metadata_dominant': {'visual': 0.2, 'ocr': 0.1, 'metadata': 0.7},
        'visual_dominant': {'visual': 0.8, 'ocr': 0.1, 'metadata': 0.1},
        'text_dominant': {'visual': 0.2, 'ocr': 0.7, 'metadata': 0.1},
        'hybrid_balanced': {'visual': 0.5, 'ocr': 0.3, 'metadata': 0.2},
        'fallback': {'visual': 0.7, 'ocr': 0.2, 'metadata': 0.1},  # Visual-heavy default
    }
    
    # ========== OCR CONFIGURATION ==========
    
    # Minimum token length for OCR matching (avoid matching very short words)
    OCR_MIN_TOKEN_LENGTH = 3
    
    # OCR matching scores
    OCR_EXACT_MATCH_SCORE = 1.0      # Full word exact match
    OCR_PARTIAL_MATCH_SCORE = 0.6    # Partial/fuzzy match
    OCR_TOKEN_BASE_SCORE = 0.3       # Base score per matching token
    
    # ========== ADAPTIVE LEARNING CONFIGURATION ==========
    
    # Enable/disable adaptive learning from user feedback
    ENABLE_ADAPTIVE_LEARNING = True
    
    # Learning parameters (more aggressive for faster adaptation)
    LEARNING_RATE = 0.25  # Faster adaptation (was 0.15)
    MIN_FEEDBACK_SAMPLES = 5  # Learn from fewer samples (was 10)
    MIN_SUCCESS_RATE = 0.2  # Lower threshold (was 0.3)
    MAX_WEIGHT_ADJUSTMENT = 0.15  # Allow bigger changes (was 0.1)
    
    # Result-level penalties
    ENABLE_RESULT_PENALTIES = True  # Penalize frequently disliked results
    
    # Feedback database path
    FEEDBACK_DB_PATH = "feedback.db"
    
    # ========== EXAMPLES ==========
    """
    Example configurations for different use cases:

    1. STRICT MODE (only nearly identical matches):
       RELATIVE_THRESHOLD = 0.15
       FLOOR_RATIO = 0.85
       MAX_RESULTS = 20

    2. BALANCED MODE (recommended default):
       RELATIVE_THRESHOLD = 0.30
       FLOOR_RATIO = 0.70
       MAX_RESULTS = 50

    3. LENIENT MODE (show more variety):
       RELATIVE_THRESHOLD = 0.40
       FLOOR_RATIO = 0.60
       MAX_RESULTS = 100

    4. EXPLORATION MODE (see everything relevant):
       RELATIVE_THRESHOLD = 0.50
       FLOOR_RATIO = 0.50
       MAX_RESULTS = 200
    """
    
    # ========== ADVANCED PARAMETERS ==========
    
    # Enable score gap detection (future feature)
    # This would detect natural gaps in score distribution
    ENABLE_GAP_DETECTION = False
    GAP_THRESHOLD = 0.05  # Minimum gap to consider as a natural break

    # ========== DUPLICATE DETECTION CONFIGURATION ==========

    # Perceptual hash (pHash) Hamming distance threshold.
    # 0  = pixel-identical images only
    # ≤10 = near-exact duplicates (JPEG saves, minor compressions)  ← recommended
    # ≤20 = looser near-duplicates (may flag similar-looking photos)
    DEDUP_PHASH_THRESHOLD = 10

    # CLIP embedding cosine similarity threshold for semantic duplicates.
    # 0.98 = near-identical rendering
    # 0.95 = very similar image (different crop/filter)           ← recommended
    # 0.90 = semantically similar (may flag related-but-distinct photos)
    DEDUP_EMBEDDING_THRESHOLD = 0.95

    # Whether to use CLIP embeddings in addition to pHash.
    # True  = catches edited/filtered duplicates pHash misses (slower)
    # False = pHash only, very fast
    DEDUP_USE_EMBEDDING = True
    
    @classmethod
    def get_config(cls):
        """Returns current configuration as a dictionary"""
        return {
            'relative_threshold': cls.RELATIVE_THRESHOLD,
            'floor_ratio': cls.FLOOR_RATIO,
            'max_results': cls.MAX_RESULTS,
            'metadata_weight': cls.METADATA_WEIGHT,
            'enable_gap_detection': cls.ENABLE_GAP_DETECTION,
            'gap_threshold': cls.GAP_THRESHOLD,
            'weight_presets': cls.WEIGHT_PRESETS,
            'ocr_min_token_length': cls.OCR_MIN_TOKEN_LENGTH,
            'ocr_exact_match_score': cls.OCR_EXACT_MATCH_SCORE,
            'ocr_partial_match_score': cls.OCR_PARTIAL_MATCH_SCORE,
            'ocr_token_base_score': cls.OCR_TOKEN_BASE_SCORE,
            'enable_adaptive_learning': cls.ENABLE_ADAPTIVE_LEARNING,
            'learning_rate': cls.LEARNING_RATE,
            'min_feedback_samples': cls.MIN_FEEDBACK_SAMPLES,
            'min_success_rate': cls.MIN_SUCCESS_RATE,
            'max_weight_adjustment': cls.MAX_WEIGHT_ADJUSTMENT,
            'feedback_db_path': cls.FEEDBACK_DB_PATH,
            'enable_result_penalties': cls.ENABLE_RESULT_PENALTIES,
            # Duplicate detection
            'dedup_phash_threshold':     cls.DEDUP_PHASH_THRESHOLD,
            'dedup_embedding_threshold': cls.DEDUP_EMBEDDING_THRESHOLD,
            'dedup_use_embedding':       cls.DEDUP_USE_EMBEDDING,
        }
    
    @classmethod
    def validate(cls):
        """Validates configuration parameters"""
        if not (0.0 <= cls.RELATIVE_THRESHOLD <= 1.0):
            raise ValueError(f"RELATIVE_THRESHOLD must be between 0.0 and 1.0, got {cls.RELATIVE_THRESHOLD}")

        if not (0.0 <= cls.FLOOR_RATIO <= 1.0):
            raise ValueError(f"FLOOR_RATIO must be between 0.0 and 1.0, got {cls.FLOOR_RATIO}")

        if cls.MAX_RESULTS < 1:
            raise ValueError(f"MAX_RESULTS must be >= 1, got {cls.MAX_RESULTS}")

        return True
