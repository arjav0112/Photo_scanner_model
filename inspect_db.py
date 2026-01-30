import sqlite3
import numpy as np
import os
import textwrap

def inspect_database(db_path="photos.db"):
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute("PRAGMA table_info(photos)")
    columns = [info[1] for info in cursor.fetchall()]
    
    print("="*100)
    print(f"DATABASE STRUCTURE: {db_path}")
    print(f"Columns: {', '.join(columns)}")
    print("="*100)

    try:
        # Check if ocr_text exists for query
        has_ocr = "ocr_text" in columns
        
        # Explicitly select columns to ensure index consistency
        # 0:id, 1:filename, 2:size, 3:path, 4:emb, 5:metadata, 6:ocr(optional)
        query_cols = "id, filename, size_bytes, path, embedding, metadata" 
        if has_ocr:
            query_cols += ", ocr_text"
        
        cursor.execute(f"SELECT {query_cols} FROM photos")
        rows = cursor.fetchall()
        
        print(f"Total Records: {len(rows)}")
        print("-" * 100)
        
        for row in rows:
            idx = row[0]
            filename = row[1]
            size_kb = row[2] / 1024 if row[2] else 0
            path = row[3]
            
            # Handle Embedding blob
            emb_blob = row[4]
            if emb_blob:
                emb_array = np.frombuffer(emb_blob, dtype=np.float32)
                emb_info = f"Blob ({len(emb_blob)} bytes) -> Vector {emb_array.shape}"
            else:
                emb_info = "None"
            
            meta_json = row[5]
            
            # OCR Text is last
            ocr_text = "N/A"
            if has_ocr and len(row) > 6:
                raw_text = row[6]
                if raw_text:
                    # Show first 100 chars
                    ocr_text = raw_text.replace('\n', ' ')
                    if len(ocr_text) > 100:
                        ocr_text = ocr_text[:100] + "..."
                else:
                    ocr_text = "[No Text Detected]"

            print(f"ID: {idx} | File: {filename} | Size: {size_kb:.1f} KB")
            print(f"Path: {path}")
            print(f"Embedding: {emb_info}")
            print(f"OCR Text: {ocr_text}")
            
            # Metadata
            try:
               import json
               if meta_json:
                   meta_dict = json.loads(meta_json)
                   print(f"Metadata: {meta_dict}")
               else:
                   print("Metadata: {}")
            except Exception as e:
               print(f"Metadata: [Error decoding] {e}")
               
            print("-" * 100)
            
    except Exception as e:
        print(f"Error reading database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_database()
