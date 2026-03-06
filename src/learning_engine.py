"""
Learning Engine Module
Implements adaptive weight learning from user feedback
"""

from typing import Dict, Optional
from src.feedback_handler import FeedbackHandler


class LearningEngine:
    """
    Learns optimal weights from user feedback using exponential moving average
    """
    
    def __init__(
        self,
        feedback_handler: FeedbackHandler,
        learning_rate: float = 0.15,
        min_samples: int = 10,
        min_success_rate: float = 0.3,
        max_adjustment: float = 0.1
    ):
        """
        Initialize learning engine
        
        Args:
            feedback_handler: Feedback storage handler
            learning_rate: How quickly to adapt (0.1-0.5, default 0.15 conservative)
            min_samples: Minimum feedback before adjusting
            min_success_rate: Minimum success rate to learn from
            max_adjustment: Maximum per-update weight change
        """
        self.feedback = feedback_handler
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.min_success_rate = min_success_rate
        self.max_adjustment = max_adjustment
    
    def update_weights(
        self,
        current_presets: Dict[str, Dict[str, float]],
        verbose: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Update weight presets based on feedback
        
        Args:
            current_presets: Current weight configuration
            verbose: Print update details
        
        Returns:
            Updated weight presets
        """
        updated_presets = current_presets.copy()
        
        intent_to_preset = {
            'metadata': 'metadata_dominant',
            'visual': 'visual_dominant',
            'text': 'text_dominant',
            'hybrid': 'hybrid_balanced',
        }
        
        for intent, preset_key in intent_to_preset.items():
            learning_data = self.feedback.get_learning_data(
                intent.upper(), 
                self.min_samples
            )
            
            if not learning_data:
                if verbose:
                    print(f"  {intent.upper()}: Insufficient data (need {self.min_samples} samples)")
                continue
            
            if learning_data['success_rate'] < self.min_success_rate:
                if verbose:
                    print(f"  {intent.upper()}: Low success rate ({learning_data['success_rate']:.2%}), skipping")
                continue
            
            current_weights = updated_presets[preset_key]
            observed_weights = learning_data['avg_successful_weights']
            
            new_weights = {}
            for component in ['visual', 'ocr', 'metadata']:
                current = current_weights[component]
                observed = observed_weights[component]
                
                suggested = current * (1 - self.learning_rate) + observed * self.learning_rate
                
                delta = suggested - current
                if abs(delta) > self.max_adjustment:
                    delta = self.max_adjustment if delta > 0 else -self.max_adjustment
                    adjusted = current + delta
                else:
                    adjusted = suggested
                
                new_weights[component] = max(0.0, adjusted)
            
            updated_presets[preset_key] = new_weights
            
            if verbose:
                print(f"  {intent.upper()}: Updated from {learning_data['total_samples']} samples "
                      f"({learning_data['success_rate']:.1%} success)")
                print(f"    Visual: {current_weights['visual']:.2f} → {new_weights['visual']:.2f}")
                print(f"    OCR: {current_weights['ocr']:.2f} → {new_weights['ocr']:.2f}")
                print(f"    Metadata: {current_weights['metadata']:.2f} → {new_weights['metadata']:.2f}")
        
        return updated_presets
    
    def should_update(self) -> bool:
        """
        Check if there's enough new feedback to trigger an update
        
        Returns:
            True if update is recommended
        """
        stats = self.feedback.get_feedback_stats()
        total = stats.get('total_feedback', 0)
        
        return total >= self.min_samples


if __name__ == "__main__":
    from search_config import SearchConfig
    
    handler = FeedbackHandler()
    engine = LearningEngine(handler)
    
    current = SearchConfig.get_config()['weight_presets']
    
    print("Testing weight updates...")
    print("="*60)
    
    updated = engine.update_weights(current, verbose=True)
    
    print("\n" + "="*60)
    print("Update complete")
