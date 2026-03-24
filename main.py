
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
    
    # Search Command
    search_parser = subparsers.add_parser("search", help="Search for photos")
    search_parser.add_argument("query", help="Text query")

    # Dedupe Command
    dedupe_parser = subparsers.add_parser("dedupe", help="Find and manage duplicate images")
    dedupe_parser.add_argument(
        "--phash-threshold", type=int, default=10,
        help="Hamming distance threshold for pHash matching (default: 10, 0=pixel-identical)"
    )
    dedupe_parser.add_argument(
        "--embedding-threshold", type=float, default=0.90,
        help="Cosine similarity threshold for embedding-based matching (default: 0.90)"
    )
    dedupe_parser.add_argument(
        "--no-embedding", action="store_true",
        help="Skip CLIP embedding similarity check (faster, pHash only)"
    )
    dedupe_parser.add_argument(
        "--auto-mark", action="store_true",
        help="Automatically mark duplicates in the database without interactive review"
    )
    dedupe_parser.add_argument(
        "--delete", action="store_true",
        help="Delete duplicate files from disk (IRREVERSIBLE — use with caution)"
    )
    dedupe_parser.add_argument(
        "--list", action="store_true",
        help="List all previously marked duplicates stored in the database"
    )

    # Face Grouping Commands
    group_faces_parser = subparsers.add_parser(
        "group-faces", help="Cluster all detected faces into person identities"
    )
    group_faces_parser.add_argument(
        "--reset", action="store_true",
        help="Wipe existing person assignments and recluster from scratch"
    )

    name_person_parser = subparsers.add_parser(
        "name-person", help="Assign a name to a person cluster"
    )
    name_person_parser.add_argument("person_id", type=int, help="Person ID from group-faces output")
    name_person_parser.add_argument("name", help="Human-readable name for this person")

    search_person_parser = subparsers.add_parser(
        "search-person", help="List all photos containing a specific person"
    )
    search_person_parser.add_argument(
        "person", help="Person name or numeric ID"
    )

    args = parser.parse_args()

    
    if args.command == "scan":
        print(f"Initializing Scanner...")
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
        
        SearchConfig.validate()
        config = SearchConfig.get_config()

        print(f"\nAnalyzing query intent...")
        analyzer = QueryAnalyzer(config['weight_presets'])
        intent, weights, debug_info = analyzer.analyze_query(args.query)
        
        print(f"Query Intent: {intent.value.upper()}")
        print(f"Weights: Visual={weights['visual']:.2f}, OCR={weights['ocr']:.2f}, Metadata={weights['metadata']:.2f}")
        print(f"Analysis: {debug_info['reason']}")
        
        ocr_tokens = analyzer.get_ocr_tokens(args.query)
        
        from src.search_cache import SearchCache
        cache = SearchCache(max_entries=50, ttl_seconds=300)
        
        text_emb = cache.get_text_embedding(args.query)
        
        if text_emb is not None:
            print(f"Disk cache hit — skipped model loading!")
            handler = None      
        else:
            print(f"Loading AI Model for Search...")
            handler = MobileCLIPHandler()
            
            print(f"Generating embedding for query: '{args.query}'")
            text_emb = handler.get_text_embedding(args.query)
            text_emb = text_emb / np.linalg.norm(text_emb)
            cache.set_text_embedding(args.query, text_emb)
        
        from src.faiss_index import FAISSIndex
        
        faiss_idx = FAISSIndex()
        faiss_idx.load_or_build(db)
        
        top_k = min(100, total_images)
        candidate_ids, vector_scores = faiss_idx.search(text_emb, top_k=top_k)
        
        print(f"FAISS returned {len(candidate_ids)} candidates from {total_images} images")
        
        candidates = db.get_batch_by_ids(candidate_ids.tolist())
        
        query_lower = args.query.lower()
        has_text_query = len(query_lower) > 2
        
        all_results = []
        
        for i, candidate in enumerate(candidates):
            v_score = float(vector_scores[i])
            
            o_score = 0.0
            if has_text_query and len(ocr_tokens) > 0:
                text = candidate['ocr_text']
                if text:
                    text_lower = text.lower()
                    token_score = 0.0
                    
                    for token in ocr_tokens:
                        if len(token) < config['ocr_min_token_length']:
                            continue
                        
                        if f" {token} " in f" {text_lower} ":
                            token_score += config['ocr_exact_match_score']
                        elif token in text_lower:
                            token_score += config['ocr_partial_match_score']
                        else:
                            ocr_words = text_lower.split()
                            for ocr_word in ocr_words:
                                if len(ocr_word) >= config['ocr_min_token_length']:
                                    if token in ocr_word or ocr_word in token:
                                        if abs(len(token) - len(ocr_word)) <= 2:
                                            token_score += config['ocr_token_base_score']
                                            break
                    
                    o_score = min(token_score, 3.0)
            
            m_scores, m_reasons = score_batch_metadata(args.query, [candidate['metadata']])
            m_score = m_scores[0]
            m_reason = m_reasons[0]
            
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
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"Compared against {total_images} images (FAISS indexed).")
        
        if config.get('enable_result_penalties', True):
            from src.feedback_handler import FeedbackHandler
            feedback_handler = FeedbackHandler(config.get('feedback_db_path', 'feedback.db'))
            
            for result in all_results:
                penalty = feedback_handler.get_result_penalty(result['path'], args.query)
                if penalty != 1.0:
                    result['original_score'] = result['score']
                    result['score'] *= penalty
                    result['penalty'] = penalty
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        SearchConfig.validate()
        config = SearchConfig.get_config()
        
        RELATIVE_THRESHOLD = config['relative_threshold']
        FLOOR_RATIO = config['floor_ratio']
        MAX_RESULTS = config['max_results']

        if len(all_results) == 0:
            top_results = []
        else:
            top_score = all_results[0]["score"]

            GAP_MIN_FRAC  = 0.10   
            MAX_GAP_SCAN  = 30     

            gap_threshold = None
            scores_to_scan = [r["score"] for r in all_results[:MAX_GAP_SCAN]]
            for j in range(1, len(scores_to_scan)):
                drop = scores_to_scan[j - 1] - scores_to_scan[j]
                if drop >= top_score * GAP_MIN_FRAC:
                    gap_threshold = scores_to_scan[j - 1] - drop * 0.05
                    break

            relative_cut = top_score * (1 - RELATIVE_THRESHOLD)
            floor_cut    = top_score * FLOOR_RATIO
            proportional_threshold = max(relative_cut, floor_cut)

            if gap_threshold is not None:
                score_threshold = max(gap_threshold, proportional_threshold)
                threshold_source = f"gap ({gap_threshold:.4f}) vs proportional ({proportional_threshold:.4f})"
            else:
                score_threshold = proportional_threshold
                threshold_source = f"proportional (no clear gap found)"
            
            top_results = [
                res for res in all_results 
                if res["score"] >= score_threshold
            ]
            
            top_results = top_results[:MAX_RESULTS]
            
            if len(top_results) == 0 and len(all_results) > 0:
                top_results = [all_results[0]]
            
            print(f"\n--- Filtering Statistics ---")
            print(f"Configuration: Relative={RELATIVE_THRESHOLD}, FloorRatio={FLOOR_RATIO}, MaxResults={MAX_RESULTS}")
            print(f"Top score: {top_score:.4f}")
            print(f"Threshold source: {threshold_source}")
            print(f"Relative cut: {relative_cut:.4f}  |  Floor cut: {floor_cut:.4f}  |  Threshold used: {score_threshold:.4f}")
            print(f"Images passing threshold: {len(top_results)}/{len(all_results)}")
            print(f"----------------------------")
        
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
                match_components = []
                if res["v_score"] > 0.2:
                    match_components.append("Visual")
                if res["o_score"] > 0:
                    match_components.append("OCR Text")
                if res["m_score"] > 0:
                    match_components.append("Metadata")
                
                match_type = " + ".join(match_components) if match_components else "Visual"
                
                feedback_info = ""
                if 'penalty' in res:
                    if res['penalty'] > 1.0:
                        feedback_info = f" (✨ Boosted {int((res['penalty'] - 1.0) * 100)}% due to positive feedback)"
                    elif res['penalty'] < 1.0:
                        feedback_info = f" (⚠ Downranked {int((1 - res['penalty']) * 100)}% due to negative feedback)"
                
                result_header = f"\n[{idx}] Score: {res['score']:.4f} ({match_type}){feedback_info}"
                result_file = f"    File: {res['path']}"
                
                score_breakdown = f"    Breakdown: Visual={res['v_score']:.3f}, OCR={res['o_score']:.1f}, Metadata={res['m_score']:.2f}"
                
                metadata_info = ""
                if res["m_reasons"]:
                    reasons_str = ", ".join(res["m_reasons"])
                    metadata_info = f"    Metadata Matches: {reasons_str}"
                
                print(result_header)
                print(result_file)
                print(score_breakdown)
                if metadata_info:
                    print(metadata_info)
                
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
                                
                                try:
                                    if sys.platform == 'win32':
                                        os.startfile(result['path'])
                                    elif sys.platform == 'darwin':
                                        subprocess.run(['open', result['path']])
                                    else:
                                        subprocess.run(['xdg-open', result['path']])
                                    
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
    
    elif args.command == "dedupe":
        import time as _time
        from src.duplicate_detector import DuplicateDetector, phash_to_bytes, compute_phash, bytes_to_phash

        db = PhotoDatabase()
        total = db.get_photo_count()
        if total == 0:
            print("No images in database. Run 'scan' first.")
            return

        # ── List mode ──────────────────────────────────────────────────────
        if args.list:
            flagged = db.get_duplicates()
            if not flagged:
                print("No duplicates currently flagged in the database.")
            else:
                print(f"\n{'='*70}")
                print(f"Flagged Duplicates ({len(flagged)} images)")
                print(f"{'='*70}")
                for d in flagged:
                    print(f"  DUPLICATE : {d['path']}")
                    print(f"  ORIGINAL  : {d['duplicate_of']}")
                    print()
            return

        # ── Detection phase ────────────────────────────────────────────────
        print(f"\nLoading {total} images from database for duplicate detection...")
        t0 = _time.time()
        entries = db.get_all_for_dedup()

        # Compute missing pHashes (existing DB records scanned before this feature)
        need_phash = [e for e in entries if e['phash'] is None]
        if need_phash:
            print(f"Computing pHash for {len(need_phash)} images (one-time backfill)...")
            from tqdm import tqdm
            for e in tqdm(need_phash, desc="pHash backfill"):
                ph = compute_phash(e['path'])
                if ph is not None:
                    db.update_phash(e['path'], phash_to_bytes(ph))
                    e['phash'] = ph

        detector = DuplicateDetector(
            phash_threshold     = args.phash_threshold,
            embedding_threshold = args.embedding_threshold,
        )

        use_embedding = not args.no_embedding
        print(f"\nRunning duplicate detection...")
        print(f"  pHash threshold      : ≤ {args.phash_threshold} bits Hamming distance")
        if use_embedding:
            print(f"  Embedding threshold  : ≥ {args.embedding_threshold:.2f} cosine similarity")
        else:
            print(f"  Embedding check      : DISABLED")

        dup_groups = detector.find_all_duplicates(entries, use_embedding=use_embedding)
        elapsed = _time.time() - t0

        print(f"\nDetection completed in {elapsed:.2f}s")
        print(f"{'='*70}")

        if not dup_groups:
            print("✅  No duplicates found.")
            return

        type_labels = {
            'exact':      '🔴 EXACT       (pixel-identical hash)',
            'near_exact': '🟠 NEAR-EXACT  (slight compression/resize)',
            'semantic':   '🟡 SEMANTIC    (visually similar, different file)',
        }

        total_dupes = sum(len(g['remove']) for g in dup_groups)
        print(f"Found {len(dup_groups)} duplicate group(s), {total_dupes} removable image(s)")
        print(f"{'='*70}")

        for gidx, group in enumerate(dup_groups, 1):
            label = type_labels.get(group['type'], group['type'])
            print(f"\nGroup {gidx} — {label}")
            print(f"  KEEP   : {group['keep']}")
            for r in group['remove']:
                print(f"  REMOVE : {r}")

        print(f"\n{'='*70}")

        # ── Auto-mark mode ─────────────────────────────────────────────────
        if args.auto_mark or args.delete:
            marked = 0
            deleted_files = 0
            for group in dup_groups:
                for r in group['remove']:
                    db.mark_as_duplicate(r, group['keep'])
                    marked += 1
                    if args.delete:
                        try:
                            if os.path.exists(r):
                                os.remove(r)
                                deleted_files += 1
                                print(f"  Deleted: {r}")
                        except OSError as e:
                            print(f"  Could not delete {r}: {e}")
            
            if args.delete:
                # Remove deleted records from DB
                to_remove = [r for g in dup_groups for r in g['remove']]
                db.delete_photos_by_path(to_remove)
                print(f"\n✅  Deleted {deleted_files} file(s) from disk and database.")
            else:
                print(f"\n✅  Marked {marked} image(s) as duplicates in the database.")
                print("   Run with --delete to remove them from disk.")
            return

        # ── Interactive review mode ────────────────────────────────────────
        print("\nInteractive Mode: Review each group before taking action.")
        print("Commands: [m] Mark as duplicate  [d] Delete from disk  [s] Skip  [q] Quit")
        print(f"{'='*70}")

        import subprocess, sys

        marked_total = 0
        deleted_total = 0

        for gidx, group in enumerate(dup_groups, 1):
            label = type_labels.get(group['type'], group['type'])
            print(f"\n[Group {gidx}/{len(dup_groups)}] {label}")
            print(f"  KEEP   : {group['keep']}")
            for ridx, r in enumerate(group['remove'], 1):
                print(f"  REMOVE {ridx}: {r}")

            while True:
                try:
                    cmd = input("  Action [m/d/s/q]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    cmd = 'q'

                if cmd == 'q':
                    print("\nExiting interactive mode.")
                    print(f"Summary: {marked_total} marked, {deleted_total} deleted.")
                    return
                elif cmd == 's':
                    print("  Skipped.")
                    break
                elif cmd == 'm':
                    for r in group['remove']:
                        db.mark_as_duplicate(r, group['keep'])
                        marked_total += 1
                    print(f"  ✓ Marked {len(group['remove'])} image(s) as duplicate.")
                    break
                elif cmd == 'd':
                    confirm = input(f"  ⚠ Delete {len(group['remove'])} file(s) from disk? [yes/no]: ").strip().lower()
                    if confirm == 'yes':
                        deleted = []
                        for r in group['remove']:
                            try:
                                if os.path.exists(r):
                                    os.remove(r)
                                    deleted.append(r)
                                    deleted_total += 1
                                    print(f"    Deleted: {r}")
                            except OSError as e:
                                print(f"    Error deleting {r}: {e}")
                        if deleted:
                            db.delete_photos_by_path(deleted)
                        break
                    else:
                        print("  Cancelled.")
                else:
                    print("  Unknown command. Use m/d/s/q.")

        print(f"\n{'='*70}")
        print(f"Done. {marked_total} image(s) marked as duplicates, {deleted_total} deleted from disk.")
    
    elif args.command == "group-faces":
        import time as _time
        db = PhotoDatabase()

        if args.reset:
            print("Resetting all person assignments...")
            db.delete_all_persons()

        print("Loading all face embeddings from database...")
        face_rows = db.get_all_faces_with_embeddings()

        if not face_rows:
            print("No faces found. Run 'scan' first so faces are detected.")
            return

        print(f"Found {len(face_rows)} face(s). Clustering...")
        t0 = _time.time()

        from src.person_clusterer import cluster_faces, compute_cluster_centroid
        import numpy as _np

        embeddings = _np.vstack([f["embedding"] for f in face_rows])
        face_ids   = [f["id"] for f in face_rows]

        label_map = cluster_faces(embeddings, face_ids)  # {face_id: cluster_label}

        # Remap cluster labels (-1 stays unassigned) → real person IDs in DB
        cluster_to_person: dict = {}
        for face_id, label in label_map.items():
            if label == -1:
                continue
            if label not in cluster_to_person:
                # Pick best-quality face as representative (highest det_score)
                pid = db.add_person(name=None, rep_face_id=face_id)
                cluster_to_person[label] = pid
            db.update_face_person(face_id, cluster_to_person[label])

        elapsed = _time.time() - t0
        persons = db.get_persons()

        print(f"\nClustering complete in {elapsed:.2f}s")
        print(f"{'='*60}")
        print(f"{'ID':<6} {'Name':<20} {'Photos with this person':>25}")
        print(f"{'='*60}")
        for p in persons:
            name = p["name"] or "(unnamed)"
            photos = db.get_photos_for_person(p["id"])
            print(f"{p['id']:<6} {name:<20} {len(photos):>25}")
        print(f"{'='*60}")
        print(f"\nUse 'name-person <id> <name>' to label any person.")
        print(f"Use 'search-person <name_or_id>' to find their photos.")

    elif args.command == "name-person":
        db = PhotoDatabase()
        db.update_person_name(args.person_id, args.name)
        print(f"Person {args.person_id} → '{args.name}'")

    elif args.command == "search-person":
        db = PhotoDatabase()
        persons = db.get_persons()

        # Match by numeric ID or name substring
        query = args.person.strip()
        matched = None
        if query.isdigit():
            pid = int(query)
            for p in persons:
                if p["id"] == pid:
                    matched = p
                    break
        else:
            q_lower = query.lower()
            for p in persons:
                if p["name"] and q_lower in p["name"].lower():
                    matched = p
                    break

        if matched is None:
            print(f"No person found matching '{query}'. Run 'group-faces' first.")
            persons = db.get_persons()
            if persons:
                print("\nKnown persons:")
                for p in persons:
                    print(f"  ID {p['id']}: {p['name'] or '(unnamed)'} — {p['face_count']} face(s)")
            return

        photos = db.get_photos_for_person(matched["id"])
        print(f"\nPerson: {matched['name'] or '(unnamed)'} (ID {matched['id']})")
        print(f"Found in {len(photos)} photo(s):\n")
        for ph in photos:
            print(f"  {ph}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()