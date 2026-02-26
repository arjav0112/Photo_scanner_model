"""
Query Intent Analysis Module
Classifies search queries and provides dynamic scoring weights
"""

from typing import Dict, Tuple, List
from enum import Enum
import re


class QueryIntent(Enum):
    """Query intent categories"""
    METADATA = "metadata"
    VISUAL = "visual"
    TEXT = "text"
    HYBRID = "hybrid"


class QueryAnalyzer:
    """
    Analyzes search queries to determine intent and calculate dynamic weights
    """
    
    # Keywords that suggest metadata-based search
    METADATA_KEYWORDS = {
        # Location
        'location', 'gps', 'place', 'city', 'country', 'state', 'town', 'village',
        'delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune',
        'india', 'coordinates', 'map',
        
        # Date/Time
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
        'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'morning', 'afternoon', 'evening', 'night',
        'recent', 'latest', 'old', 'older', 'yesterday', 'today', 'last week',
        'date', 'time', 'when', 'year', 'month', 'day',
        
        # Device/Camera
        'oneplus', 'samsung', 'iphone', 'apple', 'pixel', 'google', 'xiaomi',
        'canon', 'nikon', 'sony', 'fujifilm', 'phone', 'mobile', 'smartphone',
        'camera', 'dslr', 'device', 'taken with', 'shot on',
        
        # Camera settings
        'iso', 'aperture', 'flash', 'exposure', 'shutter', 'focal',
        
        # Image properties
        'resolution', 'dimensions', 'width', 'height', 'size', 'megapixel',
        'vertical', 'horizontal', 'portrait', 'landscape', 'square',
        'hd', 'quality', 'large', 'small',
        
        # Altitude
        'altitude', 'elevation', 'mountain', 'hill', 'peak', 'summit', 'sea level',
        
        # Face/People detection
        'face', 'faces', 'selfie', 'group', 'group photo', 'people',
        'crowd', 'team', 'family', 'duo', 'nobody', 'no people',
        
        # Scene classification
        'indoor', 'outdoor', 'outside', 'inside', 'room', 'interior',
        'nature', 'natural', 'urban',
        
        # Color analysis
        'warm', 'cool', 'vibrant', 'colorful', 'muted', 'saturated', 'vivid',
        'faded', 'pastel', 'warm tone', 'cool tone',
        
        # Quality assessment
        'sharp', 'blurry', 'blur', 'focused', 'crisp', 'clear',
        'high quality', 'low quality', 'overexposed', 'underexposed',
        'well exposed',
    }
    
    # Keywords that suggest visual content search
    VISUAL_KEYWORDS = {
        # Common objects
        'cat', 'dog', 'person', 'people', 'man', 'woman', 'child', 'baby',
        'car', 'bike', 'vehicle', 'building', 'house', 'tree', 'flower',
        'food', 'plate', 'cup', 'drink', 'animal', 'bird', 'fish',
        
        # Scenes
        'temple', 'church', 'mosque', 'beach', 'ocean', 'sea', 'river', 'lake',
        'mountain', 'forest', 'park', 'garden', 'street', 'road', 'bridge',
        'sunset', 'sunrise', 'sky', 'cloud', 'water', 'snow', 'rain',
        
        # Colors
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black',
        'white', 'brown', 'gray', 'colorful', 'bright', 'dark',
        
        # Scene descriptors
        'beside', 'near', 'behind', 'front', 'inside', 'outside',
        'beautiful', 'scenic', 'amazing', 'stunning', 'nature',
    }
    
    # Keywords that suggest OCR/text search
    TEXT_KEYWORDS = {
        'screenshot', 'text', 'document', 'sign', 'signboard', 'board',
        'caption', 'quote', 'message', 'chat', 'email', 'letter',
        'note', 'writing', 'written', 'words', 'label', 'tag',
        'menu', 'list', 'receipt', 'ticket', 'poster', 'banner',
        'error', 'notification', 'alert', 'warning',
    }
    
    def __init__(self, config: Dict = None, use_adaptive_weights: bool = True):
        """
        Initialize query analyzer
        
        Args:
            config: Configuration dictionary with weight presets
            use_adaptive_weights: Load learned weights from feedback
        """
        self.config = config or self._get_default_config()
        self.use_adaptive = use_adaptive_weights
        
        # Load learned weights if adaptive learning is enabled
        if self.use_adaptive:
            try:
                learned_config = self._load_learned_weights()
                if learned_config:
                    # Merge learned weights into config
                    self.config.update(learned_config)
            except Exception as e:
                # Silently fail if learning not available yet
                pass
    
    def _get_default_config(self) -> Dict:
        """Get default weight configuration"""
        return {
            'metadata_dominant': {'visual': 0.2, 'ocr': 0.1, 'metadata': 0.7},
            'visual_dominant': {'visual': 0.8, 'ocr': 0.1, 'metadata': 0.1},
            'text_dominant': {'visual': 0.2, 'ocr': 0.7, 'metadata': 0.1},
            'hybrid_balanced': {'visual': 0.5, 'ocr': 0.3, 'metadata': 0.2},
            'fallback': {'visual': 0.7, 'ocr': 0.2, 'metadata': 0.1},  # Visual-heavy default
        }
    
    def _load_learned_weights(self) -> Dict:
        """
        Load learned weights from feedback database
        
        Returns:
            Dict with learned weight presets or empty dict
        """
        try:
            from src.feedback_handler import FeedbackHandler
            from src.learning_engine import LearningEngine
            from search_config import SearchConfig
            
            config = SearchConfig.get_config()
            
            # Check if adaptive learning is enabled
            if not config.get('enable_adaptive_learning', False):
                return {}
            
            # Initialize feedback handler and learning engine
            handler = FeedbackHandler(config.get('feedback_db_path', 'feedback.db'))
            engine = LearningEngine(
                handler,
                learning_rate=config.get('learning_rate', 0.15),
                min_samples=config.get('min_feedback_samples', 10),
                min_success_rate=config.get('min_success_rate', 0.3),
                max_adjustment=config.get('max_weight_adjustment', 0.1)
            )
            
            # Get current presets and update with learned weights
            current_presets = config.get('weight_presets', self._get_default_config())
            updated_presets = engine.update_weights(current_presets, verbose=False)
            
            return updated_presets
            
        except ImportError:
            # Modules not available yet
            return {}
        except Exception:
            # Any other error, use defaults
            return {}
    
    def analyze_query(self, query: str) -> Tuple[QueryIntent, Dict[str, float], Dict]:
        """
        Analyze query to determine intent and calculate weights
        
        Args:
            query: Search query string
        
        Returns:
            (intent, weights, debug_info): Intent type, weight dict, and debug information
        """
        query_lower = query.lower()
        tokens = query_lower.split()
        
        # Count keyword matches for each category
        metadata_count = sum(1 for token in tokens if token in self.METADATA_KEYWORDS)
        visual_count = sum(1 for token in tokens if token in self.VISUAL_KEYWORDS)
        text_count = sum(1 for token in tokens if token in self.TEXT_KEYWORDS)
        
        # Also check for year patterns (metadata indicator)
        year_pattern = r'\b(19|20)\d{2}\b'
        if re.search(year_pattern, query):
            metadata_count += 1
        
        # Check for dimension patterns like "1920x1080" (metadata indicator)
        dimension_pattern = r'\d+x\d+'
        if re.search(dimension_pattern, query):
            metadata_count += 1
        
        # Determine intent
        total_matches = metadata_count + visual_count + text_count
        
        debug_info = {
            'metadata_matches': metadata_count,
            'visual_matches': visual_count,
            'text_matches': text_count,
            'total_tokens': len(tokens),
        }
        
        if total_matches == 0:
            # No clear intent - use fallback (visual-heavy)
            intent = QueryIntent.VISUAL
            weights = self.config['fallback'].copy()
            debug_info['reason'] = 'No matches - using visual-heavy fallback'
        
        elif metadata_count > 0 and visual_count == 0 and text_count == 0:
            # Pure metadata query
            intent = QueryIntent.METADATA
            weights = self.config['metadata_dominant'].copy()
            debug_info['reason'] = 'Pure metadata query'
        
        elif visual_count > 0 and metadata_count == 0 and text_count == 0:
            # Pure visual query
            intent = QueryIntent.VISUAL
            weights = self.config['visual_dominant'].copy()
            debug_info['reason'] = 'Pure visual query'
        
        elif text_count > 0 and visual_count == 0 and metadata_count == 0:
            # Pure OCR/text query
            intent = QueryIntent.TEXT
            weights = self.config['text_dominant'].copy()
            debug_info['reason'] = 'Pure text/OCR query'
        
        else:
            # Hybrid query - boost both components equally
            intent = QueryIntent.HYBRID
            
            # Calculate proportional weights based on matches
            if metadata_count > 0 and visual_count > 0 and text_count == 0:
                # Metadata + Visual
                total = metadata_count + visual_count
                metadata_ratio = metadata_count / total
                visual_ratio = visual_count / total
                weights = {
                    'visual': 0.4 + (visual_ratio * 0.3),
                    'ocr': 0.1,
                    'metadata': 0.4 + (metadata_ratio * 0.3),
                }
                debug_info['reason'] = 'Hybrid: metadata + visual'
            
            elif visual_count > 0 and text_count > 0 and metadata_count == 0:
                # Visual + Text
                total = visual_count + text_count
                visual_ratio = visual_count / total
                text_ratio = text_count / total
                weights = {
                    'visual': 0.4 + (visual_ratio * 0.3),
                    'ocr': 0.4 + (text_ratio * 0.3),
                    'metadata': 0.1,
                }
                debug_info['reason'] = 'Hybrid: visual + text'
            
            elif metadata_count > 0 and text_count > 0 and visual_count == 0:
                # Metadata + Text
                total = metadata_count + text_count
                metadata_ratio = metadata_count / total
                text_ratio = text_count / total
                weights = {
                    'visual': 0.1,
                    'ocr': 0.4 + (text_ratio * 0.3),
                    'metadata': 0.4 + (metadata_ratio * 0.3),
                }
                debug_info['reason'] = 'Hybrid: metadata + text'
            
            else:
                # All three or complex hybrid - use balanced
                weights = self.config['hybrid_balanced'].copy()
                debug_info['reason'] = 'Hybrid: all components'
        
        # Ensure weights sum to reasonable range (informational normalization)
        # Note: Weights don't need to sum to 1.0 since they're multiplicative factors
        
        return intent, weights, debug_info
    
    def update_config(self, new_config: Dict):
        """Update weight configuration (for adaptive learning)"""
        self.config.update(new_config)
    
    def get_ocr_tokens(self, query: str) -> List[str]:
        """
        Extract meaningful tokens from query for OCR matching
        Filters out stop words and common filler terms
        
        Args:
            query: Search query
            
        Returns:
            List of meaningful tokens for OCR matching
        """
        # Common stop words to exclude from OCR matching
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'of', 'at', 'by', 'for', 'with',
            'about', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
            # Photography-specific filler words
            'photo', 'photos', 'picture', 'pictures', 'image', 'images',
            'taken', 'shot', 'showing', 'show', 'display'
        }
        
        # Tokenize and filter
        tokens = query.lower().split()
        meaningful_tokens = [
            token for token in tokens 
            if token not in stop_words and len(token) >= 3
        ]
        
        return meaningful_tokens if meaningful_tokens else tokens


def analyze_query(query: str, config: Dict = None) -> Tuple[QueryIntent, Dict[str, float], Dict]:
    """
    Convenience function to analyze a query
    
    Args:
        query: Search query string
        config: Optional configuration dictionary
    
    Returns:
        (intent, weights, debug_info)
    """
    analyzer = QueryAnalyzer(config)
    return analyzer.analyze_query(query)
