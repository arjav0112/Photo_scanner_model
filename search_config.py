"""
Search Configuration
Adjust these parameters to control search result filtering
"""

class SearchConfig:
    """
    Configuration for adaptive threshold filtering in search results.
    
    The filtering works as follows:
    1. All images are scored based on similarity to the search query
    2. Only images with scores close to the top score are shown
    3. The threshold is calculated as: max(top_score * (1 - RELATIVE_THRESHOLD), MIN_ABSOLUTE_SCORE)
    """
    
    # ========== MAIN PARAMETERS (Adjust these) ==========
    
    # Relative Threshold: Controls how close to the top score images must be
    # - Lower values (0.05-0.10): Very strict, only nearly identical matches
    # - Medium values (0.15-0.20): Balanced, shows relevant images
    # - Higher values (0.25-0.35): More lenient, shows more variety
    # Recommended range: 0.10 to 0.30
    RELATIVE_THRESHOLD = 0.15
    
    # Minimum Absolute Score: Filters out very poor matches regardless of top score
    # - If top score is low, this prevents showing irrelevant images
    # - Set to 0.0 to disable absolute filtering
    # Recommended range: 0.15 to 0.30
    MIN_ABSOLUTE_SCORE = 0.2
    
    # Maximum Results: Safety limit to prevent overwhelming output
    # - Even if many images pass the threshold, limit to this number
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
    
    1. STRICT MODE (only very similar images):
       RELATIVE_THRESHOLD = 0.08
       MIN_ABSOLUTE_SCORE = 0.30
       MAX_RESULTS = 20
    
    2. BALANCED MODE (recommended default):
       RELATIVE_THRESHOLD = 0.15
       MIN_ABSOLUTE_SCORE = 0.20
       MAX_RESULTS = 50
    
    3. LENIENT MODE (show more variety):
       RELATIVE_THRESHOLD = 0.25
       MIN_ABSOLUTE_SCORE = 0.15
       MAX_RESULTS = 100
    
    4. EXPLORATION MODE (see everything relevant):
       RELATIVE_THRESHOLD = 0.35
       MIN_ABSOLUTE_SCORE = 0.10
       MAX_RESULTS = 200
    """
    
    # ========== ADVANCED PARAMETERS ==========
    
    # Enable score gap detection (future feature)
    # This would detect natural gaps in score distribution
    ENABLE_GAP_DETECTION = False
    GAP_THRESHOLD = 0.05  # Minimum gap to consider as a natural break
    
    @classmethod
    def get_config(cls):
        """Returns current configuration as a dictionary"""
        return {
            'relative_threshold': cls.RELATIVE_THRESHOLD,
            'min_absolute_score': cls.MIN_ABSOLUTE_SCORE,
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
        }
    
    @classmethod
    def validate(cls):
        """Validates configuration parameters"""
        if not (0.0 <= cls.RELATIVE_THRESHOLD <= 1.0):
            raise ValueError(f"RELATIVE_THRESHOLD must be between 0.0 and 1.0, got {cls.RELATIVE_THRESHOLD}")
        
        if cls.MIN_ABSOLUTE_SCORE < 0.0:
            raise ValueError(f"MIN_ABSOLUTE_SCORE must be >= 0.0, got {cls.MIN_ABSOLUTE_SCORE}")
        
        if cls.MAX_RESULTS < 1:
            raise ValueError(f"MAX_RESULTS must be >= 1, got {cls.MAX_RESULTS}")
        
        return True
