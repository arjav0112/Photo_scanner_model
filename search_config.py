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
    # - Visual similarity and OCR have weight 1.0 each
    # - Metadata weight determines relative importance of metadata matches
    # - Lower values (0.3-0.5): Metadata is a tiebreaker
    # - Medium values (0.5-0.8): Metadata significantly boosts relevant images
    # - Higher values (0.8-1.5): Metadata can override visual similarity
    # Recommended range: 0.3 to 1.0
    METADATA_WEIGHT = 0.5
    
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
            'gap_threshold': cls.GAP_THRESHOLD
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
