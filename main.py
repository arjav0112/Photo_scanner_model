
import os
import argparse
from src.scanner import PhotoScanner
from src.database import PhotoDatabase
from src.model_handler import MobileCLIPHandler
from src.metadata_scorer import score_batch_metadata
from src.query_analyzer import QueryAnalyzer
from search_config import SearchConfig
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Local Photo Scanner & Search")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan Command
    scan_parser = subparsers.add_parser("scan", help="Scan a directory for photos")
    scan_parser.add_argument("directory", help="Path to the directory to scan")
    
    # Search Command (Placeholder for now)
    search_parser = subparsers.add_parser("search", help="Search for photos (Not fully implemented without Text Tokenizer)")
    search_parser.add_argument("query", help="Text query")

    args = parser.parse_args()
    
    if args.command == "scan":
        print(f"Initializing Scanner...")
        # Model path is now ignored inside class, handled by SentenceTransformers
        scanner = PhotoScanner(model_path="")
        scanner.scan_directory(args.directory)
        print("Scan complete.")
        
    elif args.command == "search":
        import time as _time
        search_start = _time.time()
        
        print("Loading database...")
        db = PhotoDatabase()
        
        total_images = db.get_photo_count()
        if total_images == 0:
            print("No images found in database. Run 'scan' first.")
            return
        
        # Load search configuration
        SearchConfig.validate()
        config = SearchConfig.get_config()
        
        # ============================================================
        # QUERY ANALYSIS - Determine intent and get dynamic weights
        # ============================================================
        print(f"\nAnalyzing query intent...")
        analyzer = QueryAnalyzer(config['weight_presets'])
        intent, weights, debug_info = analyzer.analyze_query(args.query)
        
        print(f"Query Intent: {intent.value.upper()}")
        print(f"Weights: Visual={weights['visual']:.2f}, OCR={weights['ocr']:.2f}, Metadata={weights['metadata']:.2f}")
        print(f"Analysis: {debug_info['reason']}")
        
        # Extract OCR tokens for enhanced matching
        ocr_tokens = analyzer.get_ocr_tokens(args.query)
        
        # ============================================================
        # DISK-BASED EMBEDDING CACHE — check BEFORE loading model
        # ============================================================
        from src.search_cache import SearchCache
        cache = SearchCache(max_entries=50, ttl_seconds=300)
        
        # Try disk cache first (no model needed!)
        text_emb = cache.get_text_embedding(args.query)
        
        if text_emb is not None:
            print(f"Disk cache hit — skipped model loading!")
            handler = None  # Model never loaded
        else:
            # Cache miss — must load model to encode new query
            print(f"Loading AI Model for Search...")
            handler = MobileCLIPHandler()
            
            print(f"Generating embedding for query: '{args.query}'")
            text_emb = handler.get_text_embedding(args.query)
            text_emb = text_emb / np.linalg.norm(text_emb)
            cache.set_text_embedding(args.query, text_emb)
        
        # ============================================================
        # FAISS SEARCH - Fast vector similarity (replaces np.dot loop)
        # ============================================================
        from src.faiss_index import FAISSIndex
        
        faiss_idx = FAISSIndex()
        faiss_idx.load_or_build(db)
        
        # Get top candidates (only these get OCR/metadata scoring)
        top_k = min(100, total_images)
        candidate_ids, vector_scores = faiss_idx.search(text_emb, top_k=top_k)
        
        print(f"FAISS returned {len(candidate_ids)} candidates from {total_images} images")
        
        # ============================================================
        # RETRIEVE CANDIDATE DATA (only top results, not all images)
        # ============================================================
        candidates = db.get_batch_by_ids(candidate_ids.tolist())
        
        # Map scores to candidates (maintain order)
        query_lower = args.query.lower()
        has_text_query = len(query_lower) > 2
        
        all_results = []
        
        for i, candidate in enumerate(candidates):
            v_score = float(vector_scores[i])
            
            # ============================================================
            # OCR SCORING (only on FAISS candidates, not all images)
            # ============================================================
            o_score = 0.0
            if has_text_query and len(ocr_tokens) > 0:
                text = candidate['ocr_text']
                if text:
                    text_lower = text.lower()
                    token_score = 0.0
                    
                    for token in ocr_tokens:
                        if len(token) < config['ocr_min_token_length']:
                            continue
                        
                        # Exact word match (highest score)
                        if f" {token} " in f" {text_lower} ":
                            token_score += config['ocr_exact_match_score']
                        # Partial/fuzzy match (medium score)
                        elif token in text_lower:
                            token_score += config['ocr_partial_match_score']
                        # Fuzzy matching
                        else:
                            ocr_words = text_lower.split()
                            for ocr_word in ocr_words:
                                if len(ocr_word) >= config['ocr_min_token_length']:
                                    if token in ocr_word or ocr_word in token:
                                        if abs(len(token) - len(ocr_word)) <= 2:
                                            token_score += config['ocr_token_base_score']
                                            break
                    
                    o_score = min(token_score, 3.0)
            
            # ============================================================
            # METADATA SCORING (only on FAISS candidates)
            # ============================================================
            m_scores, m_reasons = score_batch_metadata(args.query, [candidate['metadata']])
            m_score = m_scores[0]
            m_reason = m_reasons[0]
            
            # Dynamic weighted scoring
            final_score = (
                v_score * weights['visual'] +
                o_score * weights['ocr'] +
                m_score * weights['metadata']
            )
            
            all_results.append({
                "path": candidate['path'],
                "score": final_score,
                "v_score": v_score,
                "o_score": o_score,
                "m_score": m_score,
                "m_reasons": m_reason
            })
        
        # Sort results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"Compared against {total_images} images (FAISS indexed).")
        
        # ============================================================
        # APPLY RESULT-LEVEL PENALTIES (Immediate Feedback Impact)
        # ============================================================
        if config.get('enable_result_penalties', True):
            from src.feedback_handler import FeedbackHandler
            feedback_handler = FeedbackHandler(config.get('feedback_db_path', 'feedback.db'))
            
            for result in all_results:
                penalty = feedback_handler.get_result_penalty(result['path'], args.query)
                if penalty != 1.0:
                    result['original_score'] = result['score']
                    result['score'] *= penalty
                    result['penalty'] = penalty
        
        # Re-sort after applying penalties
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # ============================================================
        # ADAPTIVE THRESHOLD FILTERING
        # Only show images with scores close to the top score
        # ============================================================
        
        # Load configuration
        SearchConfig.validate()  # Ensure config is valid
        config = SearchConfig.get_config()
        
        RELATIVE_THRESHOLD = config['relative_threshold']
        MIN_ABSOLUTE_SCORE = config['min_absolute_score']
        MAX_RESULTS = config['max_results']
        
        if len(all_results) == 0:
            top_results = []
        else:
            top_score = all_results[0]["score"]
            
            # Calculate dynamic threshold based on top score
            # Formula: threshold = top_score * (1 - RELATIVE_THRESHOLD)
            # This means we show images with score >= (top_score - top_score * RELATIVE_THRESHOLD)
            score_threshold = max(
                top_score * (1 - RELATIVE_THRESHOLD),  # Relative threshold
                MIN_ABSOLUTE_SCORE                      # Absolute minimum
            )
            
            # Filter results based on threshold
            top_results = [
                res for res in all_results 
                if res["score"] >= score_threshold
            ]
            
            # Apply safety limit
            top_results = top_results[:MAX_RESULTS]
            
            # Ensure at least the top result is shown (even if below absolute threshold)
            if len(top_results) == 0 and len(all_results) > 0:
                top_results = [all_results[0]]
            
            # Log filtering statistics
            print(f"\n--- Filtering Statistics ---")
            print(f"Configuration: Relative={RELATIVE_THRESHOLD}, MinScore={MIN_ABSOLUTE_SCORE}, MaxResults={MAX_RESULTS}")
            print(f"Top score: {top_score:.4f}")
            print(f"Score threshold: {score_threshold:.4f}")
            print(f"Images passing threshold: {len(top_results)}/{len(all_results)}")
            print(f"----------------------------")
        
        # Capture output
        with open("search_results.log", "w", encoding="utf-8") as log:
            log.write("Search Results:\n")
            log.write(f"Query: '{args.query}'\n")
            log.write(f"Query Intent: {intent.value.upper()}\n")
            log.write(f"Dynamic Weights: Visual={weights['visual']:.2f}, OCR={weights['ocr']:.2f}, Metadata={weights['metadata']:.2f}\n")
            log.write(f"Analysis: {debug_info['reason']}\n")
            log.write(f"Total images found: {len(top_results)}\n")
            log.write("="*80 + "\n\n")
            
            print("\nTop Matches:")
            print("="*80)
            print(f"Intent: {intent.value.upper()} | Weights: V={weights['visual']:.2f} O={weights['ocr']:.2f} M={weights['metadata']:.2f}")
            print("="*80)
            
            for idx, res in enumerate(top_results, 1):
                # Determine match type
                match_components = []
                if res["v_score"] > 0.2:
                    match_components.append("Visual")
                if res["o_score"] > 0:
                    match_components.append("OCR Text")
                if res["m_score"] > 0:
                    match_components.append("Metadata")
                
                match_type = " + ".join(match_components) if match_components else "Visual"
                
                # Check if result was penalized or boosted
                feedback_info = ""
                if 'penalty' in res:
                    if res['penalty'] > 1.0:
                        feedback_info = f" (✨ Boosted {int((res['penalty'] - 1.0) * 100)}% due to positive feedback)"
                    elif res['penalty'] < 1.0:
                        feedback_info = f" (⚠ Downranked {int((1 - res['penalty']) * 100)}% due to negative feedback)"
                
                # Format output
                result_header = f"\n[{idx}] Score: {res['score']:.4f} ({match_type}){feedback_info}"
                result_file = f"    File: {res['path']}"
                
                # Score breakdown
                score_breakdown = f"    Breakdown: Visual={res['v_score']:.3f}, OCR={res['o_score']:.1f}, Metadata={res['m_score']:.2f}"
                
                # Metadata reasons
                metadata_info = ""
                if res["m_reasons"]:
                    reasons_str = ", ".join(res["m_reasons"])
                    metadata_info = f"    Metadata Matches: {reasons_str}"
                
                # Print to console
                print(result_header)
                print(result_file)
                print(score_breakdown)
                if metadata_info:
                    print(metadata_info)
                
                # Write to log
                log.write(result_header + "\n")
                log.write(result_file + "\n")
                log.write(score_breakdown + "\n")
                if metadata_info:
                    log.write(metadata_info + "\n")
                log.write("\n")
            
            print("="*80)
            search_elapsed = _time.time() - search_start
            print(f"\nSearch completed in {search_elapsed:.2f}s")
            print(f"Results saved to: search_results.log")
            
            # ============================================================
            # INTERACTIVE FEEDBACK COLLECTION
            # ============================================================
            if len(top_results) > 0:
                print("\n" + "="*80)
                print("FEEDBACK MODE - Help improve search results!")
                print("="*80)
                print("Commands:")
                print("  [1-9]  View image (opens in default viewer)")
                print("  [y]    Mark last viewed result as helpful")
                print("  [n]    Mark last viewed result as not helpful")
                print("  [s]    Show feedback statistics")
                print("  [q]    Quit without more feedback")
                print("="*80)
                
                from src.feedback_handler import FeedbackHandler, FeedbackType
                import subprocess
                import sys
                
                feedback_handler = FeedbackHandler(config.get('feedback_db_path', 'feedback.db'))
                last_viewed_idx = None
                
                while True:
                    try:
                        user_input = input("\nYour choice: ").strip().lower()
                        
                        if user_input == 'q':
                            print("Thanks for your feedback!")
                            break
                        
                        elif user_input == 's':
                            # Show statistics
                            stats = feedback_handler.get_feedback_stats()
                            print(f"\n--- Feedback Statistics ---")
                            print(f"Total feedback: {stats['total_feedback']}")
                            if stats['by_intent']:
                                print(f"By intent: {stats['by_intent']}")
                            if stats['by_type']:
                                print(f"By type: {stats['by_type']}")
                            print("---------------------------")
                        
                        elif user_input.isdigit():
                            idx = int(user_input) - 1
                            if 0 <= idx < len(top_results):
                                result = top_results[idx]
                                print(f"\nOpening: {result['path']}")
                                
                                # Open image in default viewer
                                try:
                                    if sys.platform == 'win32':
                                        os.startfile(result['path'])
                                    elif sys.platform == 'darwin':
                                        subprocess.run(['open', result['path']])
                                    else:
                                        subprocess.run(['xdg-open', result['path']])
                                    
                                    # Record implicit positive feedback (clicked)
                                    feedback_handler.record_feedback(
                                        query=args.query,
                                        query_intent=intent.value.upper(),
                                        weights_used=weights,
                                        result_path=result['path'],
                                        result_rank=idx + 1,
                                        result_scores={
                                            'visual': float(result['v_score']),
                                            'ocr': float(result['o_score']),
                                            'metadata': float(result['m_score']),
                                            'final': float(result['score'])
                                        },
                                        feedback_type=FeedbackType.CLICKED
                                    )
                                    
                                    last_viewed_idx = idx
                                    print("✓ Feedback recorded (CLICKED)")
                                    
                                except Exception as e:
                                    print(f"Error opening image: {e}")
                            else:
                                print(f"Invalid number. Choose 1-{len(top_results)}")
                        
                        elif user_input == 'y':
                            if last_viewed_idx is not None:
                                result = top_results[last_viewed_idx]
                                feedback_handler.record_feedback(
                                    query=args.query,
                                    query_intent=intent.value.upper(),
                                    weights_used=weights,
                                    result_path=result['path'],
                                    result_rank=last_viewed_idx + 1,
                                    result_scores={
                                        'visual': float(result['v_score']),
                                        'ocr': float(result['o_score']),
                                        'metadata': float(result['m_score']),
                                        'final': float(result['score'])
                                    },
                                    feedback_type=FeedbackType.POSITIVE
                                )
                                print("✓ Marked as HELPFUL - thanks!")
                            else:
                                print("Please view an image first (use 1-9)")
                        
                        elif user_input == 'n':
                            if last_viewed_idx is not None:
                                result = top_results[last_viewed_idx]
                                feedback_handler.record_feedback(
                                    query=args.query,
                                    query_intent=intent.value.upper(),
                                    weights_used=weights,
                                    result_path=result['path'],
                                    result_rank=last_viewed_idx + 1,
                                    result_scores={
                                        'visual': float(result['v_score']),
                                        'ocr': float(result['o_score']),
                                        'metadata': float(result['m_score']),
                                        'final': float(result['score'])
                                    },
                                    feedback_type=FeedbackType.NEGATIVE
                                )
                                print("✓ Marked as NOT HELPFUL - we'll learn from this!")
                            else:
                                print("Please view an image first (use 1-9)")
                        
                        else:
                            print("Invalid command. Use: 1-9, y, n, s, or q")
                    
                    except KeyboardInterrupt:
                        print("\n\nExiting feedback mode...")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
