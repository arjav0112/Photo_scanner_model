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
                   print(f"\n  [Metadata]:")
                   
                   # Image Properties
                   if any(k in meta_dict for k in ['width', 'height', 'format', 'mode']):
                       print(f"    Image: {meta_dict.get('width', '?')}x{meta_dict.get('height', '?')} | "
                             f"Format: {meta_dict.get('format', 'N/A')} | Mode: {meta_dict.get('mode', 'N/A')}")
                   
                   # Device Information
                   device_info = []
                   if 'device_make' in meta_dict:
                       device_info.append(meta_dict['device_make'])
                   if 'device_model' in meta_dict:
                       device_info.append(meta_dict['device_model'])
                   if device_info:
                       print(f"    Device: {' '.join(device_info)}")
                   if 'software' in meta_dict:
                       print(f"    Software: {meta_dict['software']}")
                   
                   # Date/Time
                   if 'date_taken' in meta_dict:
                       print(f"    Date Taken: {meta_dict['date_taken']}")
                   
                   # Camera Settings
                   camera_settings = []
                   if 'iso' in meta_dict:
                       camera_settings.append(f"ISO {meta_dict['iso']}")
                   if 'aperture' in meta_dict:
                       camera_settings.append(meta_dict['aperture'])
                   if 'shutter_speed' in meta_dict:
                       camera_settings.append(meta_dict['shutter_speed'])
                   if 'focal_length' in meta_dict:
                       camera_settings.append(meta_dict['focal_length'])
                   if camera_settings:
                       print(f"    Camera: {' | '.join(camera_settings)}")
                   if 'flash' in meta_dict:
                       print(f"    Flash: {meta_dict['flash']}")
                   
                   # GPS Location
                   if 'location' in meta_dict:
                       print(f"    [GPS]: {meta_dict['location']}")
                       if 'location_name' in meta_dict:
                           print(f"    [Location]: {meta_dict['location_name']}")
                       if 'gps_altitude' in meta_dict:
                           print(f"    Altitude: {meta_dict['gps_altitude']}")
                       if 'gps_timestamp' in meta_dict:
                           print(f"    GPS Time: {meta_dict['gps_timestamp']}")
                   
                   # Show raw if empty
                   if not meta_dict:
                       print(f"    (No metadata available)")
               else:
                   print(f"\n  [Metadata]: (None)")
            except Exception as e:
               print(f"\n  [Metadata]: [Error decoding] {e}")
               
            print("-" * 100)
            
    except Exception as e:
        print(f"Error reading database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_database()
