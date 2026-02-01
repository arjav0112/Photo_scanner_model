"""Check if EXIF contains address data"""
from PIL import Image
from PIL.ExifTags import TAGS
import os

test_dir = r"C:\Users\LENOVO\OneDrive\Desktop\Photo_scanner\test_images"
images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

if images:
    test_path = os.path.join(test_dir, images[0])
    print(f"Checking EXIF data in: {images[0]}\n")
    print("=" * 80)
    
    with Image.open(test_path) as img:
        exif = img._getexif()
        if exif:
            print("All EXIF Tags:\n")
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                # Look for location/address related tags
                if 'location' in str(tag_name).lower() or 'address' in str(tag_name).lower() or 'place' in str(tag_name).lower():
                    print(f"  *** {tag_name}: {value}")
                elif tag_name in ['UserComment', 'ImageDescription', 'XPComment', 'XPKeywords']:
                    print(f"  {tag_name}: {value}")
            
            print("\n" + "=" * 80)
            print("\nSearching for address in common tags...")
            
            # Check specific tags that might contain address
            address_tags = ['UserComment', 'ImageDescription', 'XPComment', 'LocationInfo']
            found_address = False
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name in address_tags and value:
                    print(f"  {tag_name}: {value}")
                    found_address = True
            
            if not found_address:
                print("  No address data found in EXIF tags")
                print("  (Address must be added manually or via online geocoding)")
        else:
            print("No EXIF data found")
