
import os
import argparse
from src.scanner import PhotoScanner
from src.database import PhotoDatabase
from src.model_handler import MobileCLIPHandler
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
        print(f"Loading AI Model for Search...")
        # Initialize robust SentenceTransformer model
        handler = MobileCLIPHandler()
        
        print(f"Generating embedding for query: '{args.query}'")
        text_emb = handler.get_text_embedding(args.query)
        
        # Normalize text embedding
        text_emb = text_emb / np.linalg.norm(text_emb)
        
        print("Loading database...")
        db = PhotoDatabase()
        
        # Batch Processing for efficient memory usage
        all_results = []
        
        batch_gen = db.get_search_data_generator(batch_size=2048) # Process in chunks
        
        query_lower = args.query.lower()
        has_text_query = len(query_lower) > 2
        
        total_images = 0
        
        for paths, image_embs, ocr_texts in batch_gen:
             count = len(paths)
             total_images += count
             
             # Normalize
             norms = np.linalg.norm(image_embs, axis=1, keepdims=True)
             image_embs = image_embs / (norms + 1e-8)
             
             # Vector Scores
             vector_scores = np.dot(image_embs, text_emb)
             
             # OCR Scores
             ocr_scores = np.zeros(count)
             if has_text_query:
                 for i, text in enumerate(ocr_texts):
                     if text and query_lower in text.lower():
                         ocr_scores[i] = 1.0
             
             # Combine
             final_batch_scores = vector_scores + ocr_scores
             
             # Collect relevant results (e.g. score > 0.1 to save memory, or all)
             # Storing all for now to exact match previous behavior
             for i in range(count):
                 all_results.append({
                     "path": paths[i],
                     "score": final_batch_scores[i],
                     "v_score": vector_scores[i],
                     "o_score": ocr_scores[i]
                 })
                 
        if total_images == 0:
            print("No images found in database. Run 'scan' first.")
            return

        print(f"Compared against {total_images} images (Batched).")
        
        # Sort results (highest similarity first)
        
        # --- CONTROL: Change this value to show more/less results ---
        top_k = 5 
        # ------------------------------------------------------------
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:top_k]
        
        # Capture output
        with open("search_results.log", "w", encoding="utf-8") as log:
            log.write("Search Results:\n")
            print("\nTop Matches:")
            for res in top_results:
                match_type = " (Visual)"
                if res["o_score"] > 0:
                    match_type = " (Text + Visual)"
                
                result_str = f"Score: {res['score']:.4f}{match_type}  |  File: {res['path']}"
                print(result_str)
                log.write(result_str + "\n")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
