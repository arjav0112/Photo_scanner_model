"""Scan directory and report which images have GPS data"""
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def check_gps(image_path):
    """Check if image has GPS data"""
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    if TAGS.get(tag, tag) == 'GPSInfo':
                        gps_data = {}
                        for gps_tag in value:
                            gps_data[GPSTAGS.get(gps_tag, gps_tag)] = value[gps_tag]
                        
                        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                            return True, gps_data
        return False, None
    except:
        return False, None

# Scan directory
test_dir = r"C:\Users\LENOVO\OneDrive\Desktop\Photo_scanner\test_images"
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]

print(f"Scanning {len(image_files)} images for GPS data...\n")
print("=" * 80)

gps_count = 0
no_gps_count = 0
sample_with_gps = []
sample_without_gps = []

for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    has_gps, gps_data = check_gps(img_path)
    
    if has_gps:
        gps_count += 1
        if len(sample_with_gps) < 5:
            sample_with_gps.append(img_name)
    else:
        no_gps_count += 1
        if len(sample_without_gps) < 5:
            sample_without_gps.append(img_name)

print(f"\n📊 RESULTS:")
print(f"  ✅ Images WITH GPS data: {gps_count} ({gps_count/len(image_files)*100:.1f}%)")
print(f"  ❌ Images WITHOUT GPS data: {no_gps_count} ({no_gps_count/len(image_files)*100:.1f}%)")

if sample_with_gps:
    print(f"\n📍 Sample images WITH GPS (first 5):")
    for name in sample_with_gps:
        print(f"  • {name}")
else:
    print(f"\n❌ No images with GPS data found in this directory")
    print(f"\n💡 To get GPS data, you need:")
    print(f"  1. Photos taken with a smartphone (with location enabled)")
    print(f"  2. Photos from a camera with GPS capability")
    print(f"  3. Original photos (not downloaded/edited)")

if sample_without_gps:
    print(f"\n📷 Sample images WITHOUT GPS (first 5):")
    for name in sample_without_gps:
        print(f"  • {name}")

print("\n" + "=" * 80)
