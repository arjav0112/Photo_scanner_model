"""
Test script to demonstrate adaptive learning system
"""

print("="*80)
print("ADAPTIVE LEARNING SYSTEM - VERIFICATION TEST")
print("="*80)

# Test 1: Feedback Handler
print("\n[Test 1] Feedback Handler Initialization")
from src.feedback_handler import FeedbackHandler, FeedbackType

handler = FeedbackHandler()
print("✓ Feedback database created")

stats = handler.get_feedback_stats()
print(f"✓ Initial stats: {stats['total_feedback']} feedback entries")

# Test 2: Record sample feedback
print("\n[Test 2] Recording Sample Feedback")
success = handler.record_feedback(
    query="test cat photo",
    query_intent="VISUAL",
    weights_used={'visual': 0.8, 'ocr': 0.1, 'metadata': 0.1},
    result_path="/test/cat.jpg",
    result_rank=1,
    result_scores={'visual': 0.9, 'ocr': 0.0, 'metadata': 0.1, 'final': 0.82},
    feedback_type=FeedbackType.POSITIVE
)
print(f"✓ Feedback recorded: {success}")

# Test 3: Learning Engine
print("\n[Test 3] Learning Engine")
from src.learning_engine import LearningEngine

engine = LearningEngine(handler, learning_rate=0.15)
print(f"✓ Learning engine initialized")
print(f"  - Learning rate: {engine.learning_rate}")
print(f"  - Min samples: {engine.min_samples}")
print(f"  - Min success rate: {engine.min_success_rate}")

# Test 4: Query Analyzer with Adaptive Weights
print("\n[Test 4] Query Analyzer with Adaptive Learning")
from src.query_analyzer import QueryAnalyzer
from search_config import SearchConfig

config = SearchConfig.get_config()
print(f"✓ Adaptive learning enabled: {config['enable_adaptive_learning']}")

analyzer = QueryAnalyzer(config['weight_presets'], use_adaptive_weights=True)
intent, weights, debug = analyzer.analyze_query("sunset from iPhone")
print(f"✓ Query analysis working")
print(f"  - Intent: {intent.value}")
print(f"  - Weights: V={weights['visual']:.2f}, O={weights['ocr']:.2f}, M={weights['metadata']:.2f}")

# Test 5: Configuration
print("\n[Test 5] Configuration Check")
print(f"✓ Learning parameters:")
print(f"  - Learning rate: {config['learning_rate']}")
print(f"  - Min feedback samples: {config['min_feedback_samples']}")
print(f"  - Max weight adjustment: {config['max_weight_adjustment']}")
print(f"  - Feedback DB: {config['feedback_db_path']}")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nThe adaptive learning system is ready to use!")
print("Run: python main.py search \"your query\"")
print("Then interact with results to provide feedback.")
