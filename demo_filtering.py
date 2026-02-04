"""
Demo script to visualize how the adaptive threshold filtering works
Run this to understand the filtering behavior without running actual searches
"""

def demonstrate_filtering(top_score, all_scores, relative_threshold, min_absolute_score, max_results):
    """
    Demonstrates the filtering logic with example scores
    """
    print(f"\n{'='*70}")
    print(f"DEMONSTRATION: Filtering with top_score={top_score}")
    print(f"Configuration: RELATIVE_THRESHOLD={relative_threshold}, MIN_ABSOLUTE_SCORE={min_absolute_score}")
    print(f"{'='*70}")
    
    # Calculate threshold
    score_threshold = max(
        top_score * (1 - relative_threshold),
        min_absolute_score
    )
    
    print(f"\nCalculated threshold: {score_threshold:.4f}")
    print(f"  Formula: max({top_score} × (1 - {relative_threshold}), {min_absolute_score})")
    print(f"  Result: max({top_score * (1 - relative_threshold):.4f}, {min_absolute_score})")
    
    # Filter results
    filtered_scores = [s for s in all_scores if s >= score_threshold]
    filtered_scores = filtered_scores[:max_results]
    
    print(f"\nResults:")
    print(f"  Total images: {len(all_scores)}")
    print(f"  Images passing threshold: {len(filtered_scores)}")
    print(f"  Percentage shown: {len(filtered_scores)/len(all_scores)*100:.1f}%")
    
    print(f"\nScore breakdown:")
    shown_count = 0
    hidden_count = 0
    
    for score in all_scores[:20]:  # Show first 20 for brevity
        if score >= score_threshold and shown_count < max_results:
            status = "✓ SHOWN"
            shown_count += 1
        else:
            status = "✗ HIDDEN"
            hidden_count += 1
        
        diff_from_top = top_score - score
        percent_diff = (diff_from_top / top_score) * 100 if top_score > 0 else 0
        
        print(f"  Score: {score:.4f} | Diff from top: {diff_from_top:.4f} ({percent_diff:.1f}%) | {status}")
    
    if len(all_scores) > 20:
        print(f"  ... and {len(all_scores) - 20} more images")


# ============================================================
# SCENARIO 1: Good match quality (searching for "cat" with many cat images)
# ============================================================
print("\n" + "="*70)
print("SCENARIO 1: High-quality matches (e.g., searching for 'cat')")
print("="*70)

top_score_1 = 0.85
scores_1 = [0.85, 0.84, 0.82, 0.80, 0.78, 0.75, 0.70, 0.65, 0.60, 0.55, 
            0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

demonstrate_filtering(
    top_score=top_score_1,
    all_scores=scores_1,
    relative_threshold=0.15,
    min_absolute_score=0.20,
    max_results=50
)

# ============================================================
# SCENARIO 2: Poor match quality (searching for "unicorn" with no unicorn images)
# ============================================================
print("\n" + "="*70)
print("SCENARIO 2: Low-quality matches (e.g., searching for 'unicorn')")
print("="*70)

top_score_2 = 0.30
scores_2 = [0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12,
            0.10, 0.08, 0.06, 0.04, 0.02]

demonstrate_filtering(
    top_score=top_score_2,
    all_scores=scores_2,
    relative_threshold=0.15,
    min_absolute_score=0.20,
    max_results=50
)

# ============================================================
# SCENARIO 3: Comparing different RELATIVE_THRESHOLD values
# ============================================================
print("\n" + "="*70)
print("SCENARIO 3: Effect of different RELATIVE_THRESHOLD values")
print("="*70)

top_score_3 = 0.80
scores_3 = [0.80, 0.78, 0.75, 0.72, 0.70, 0.68, 0.65, 0.62, 0.60, 0.55,
            0.50, 0.45, 0.40, 0.35, 0.30]

print("\n--- With STRICT threshold (0.08) ---")
demonstrate_filtering(top_score_3, scores_3, 0.08, 0.20, 50)

print("\n--- With BALANCED threshold (0.15) ---")
demonstrate_filtering(top_score_3, scores_3, 0.15, 0.20, 50)

print("\n--- With LENIENT threshold (0.25) ---")
demonstrate_filtering(top_score_3, scores_3, 0.25, 0.20, 50)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Key Insights:

1. RELATIVE_THRESHOLD controls the "tightness" of filtering:
   - Lower values (0.08-0.10) = Only very similar images
   - Medium values (0.15-0.20) = Balanced, relevant images
   - Higher values (0.25-0.35) = More variety

2. MIN_ABSOLUTE_SCORE acts as a safety net:
   - Prevents showing poor matches even if they're close to a low top score
   - Example: If top score is 0.30, images with 0.25 won't be shown if MIN=0.20

3. The formula adapts to match quality:
   - Good matches (high top score) → More images shown
   - Poor matches (low top score) → Fewer images shown (protected by MIN_ABSOLUTE_SCORE)

4. Practical usage:
   - Start with defaults (RELATIVE=0.15, MIN=0.20)
   - Adjust based on your needs:
     * Too many results? Decrease RELATIVE or increase MIN
     * Too few results? Increase RELATIVE or decrease MIN
""")

print("\nTo adjust these values, edit search_config.py")
print("="*70)
