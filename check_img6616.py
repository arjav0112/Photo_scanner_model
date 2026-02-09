"""Check EXIF data for IMG_6616.jpeg"""
from PIL import Image
from PIL.ExifTags import TAGS
import os

img_path = r"C:\Users\LENOVO\OneDrive\Desktop\Photo_scanner\test_images\IMG_6616.jpeg"

if os.path.exists(img_path):
    print(f"Checking: {img_path}\n")
    print("="*80)
    
    with Image.open(img_path) as img:
        exif = img._getexif()
        if exif:
            print("EXIF Data:\n")
            for tag, value in sorted(exif.items()):
                tag_name = TAGS.get(tag, tag)
                # Print Make, Model, and other device-related tags
                if tag_name in ['Make', 'Model', 'LensMake', 'LensModel', 'Software']:
                    print(f"  **{tag_name}**: {value}")
                elif tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                    print(f"  {tag_name}: {value}")
                elif 'GPS' in str(tag_name):
                    print(f"  {tag_name}: {value}")
        else:
            print("No EXIF data found!")
else:
    print(f"File not found: {img_path}")
