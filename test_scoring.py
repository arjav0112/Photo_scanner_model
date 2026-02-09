from src.database import PhotoDatabase
from src.metadata_scorer import score_batch_metadata
import json
import sqlite3

# Initialize database
db = PhotoDatabase()

# Get all photos
conn = sqlite3.connect(db.db_path)
cursor = conn.execute('SELECT path, metadata FROM photos')
all_photos = cursor.fetchall()
conn.close()

print(f"Total photos in DB: {len(all_photos)}\n")

# Test query
query = "Parrot photo taken from iphone"
print(f"Query: '{query}'\n")
print("="*80)

# Score each photo's metadata
for path, metadata_json in all_photos:
    metadata = json.loads(metadata_json) if metadata_json else {}
    device = metadata.get('device', 'N/A')
    
    # Score this metadata
    scores, reasons_list = score_batch_metadata(query, [metadata])
    meta_score = scores[0]
    meta_reasons = reasons_list[0]
    
    print(f"\nFile: {path}")
    print(f"  Device: {device}")
    print(f"  Metadata Score: {meta_score:.2f}")
    if meta_reasons:
        print(f"  Reasons: {', '.join(meta_reasons)}")
print("\n" + "="*80)
