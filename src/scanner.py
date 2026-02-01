
import os
import time
from tqdm import tqdm
from .database import PhotoDatabase
from .model_handler import MobileCLIPHandler
import mimetypes
import numpy as np

# Offline reverse geocoding for GPS coordinates
try:
    import reverse_geocoder as rg
    GEOCODER_AVAILABLE = True
except ImportError:
    GEOCODER_AVAILABLE = False


class PhotoScanner:
    def __init__(self, model_path: str, db_path: str = "photos.db"):
        self.db = PhotoDatabase(db_path)
        print("Loading AI Model (this may take a moment)...")
        # Initialize robust SentenceTransformer model
        self.model = MobileCLIPHandler()
        
        # Initialize OCR (lazy load later or now)
        from .ocr_handler import OCRHandler
        self.ocr = OCRHandler()
        
        # Pre-compute document anchor embedding for Gatekeeper
        print("Initializing Gatekeeper (Document Detection)...")
        doc_text = "text document invoice receipt book page letter contract"
        self.doc_embedding = self.model.get_text_embedding(doc_text)
        self.doc_embedding /= np.linalg.norm(self.doc_embedding) # Normalize
        self.doc_threshold = 0.17 # Lowered from 0.21 based on experimentation

        # Supported extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def scan_directory(self, directory: str):
        """Recursively scans a directory for photos."""
        print(f"Scanning directory: {directory}")
        
        # Get set of already scanned paths
        scanned_paths = self.db.get_scanned_paths()
        
        # Collect new files
        new_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    full_path = os.path.abspath(os.path.join(root, file))
                    if full_path not in scanned_paths:
                        new_files.append(full_path)
        
        if not new_files:
            print("No new photos found.")
            return

        print(f"Found {len(new_files)} new photos. Processing in batches...")
        
        batch_size = 16 # Adjust based on RAM
        
        # Generator for batches
        def chunked(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        for batch_idx, batch_paths in enumerate(tqdm(chunked(new_files, batch_size), total=len(new_files)//batch_size + 1, desc="Scanning Batches")):
            try:
                if batch_idx % 5 == 0: # Log every 5 batches
                     self.model.log_memory(f"Scanning Batch {batch_idx}")
                     
                # 1. Batch Embeddings (The Slow Part Optimized)
                embeddings = self.model.get_image_embeddings_batch(batch_paths)
                
                if embeddings is None:
                    continue
                
                # Process each result in batch
                for i, path in enumerate(batch_paths):
                    embedding = embeddings[i]
                    # Skip flat zero embeddings (failed loads)
                    if np.all(embedding == 0):
                        continue
                        
                    # Normalize
                    emb_norm = embedding / np.linalg.norm(embedding)
                    
                    # 2. Gatekeeper
                    sim_score = np.dot(emb_norm, self.doc_embedding)
                    
                    ocr_text = ""
                    if sim_score > self.doc_threshold:
                         # Keep visual feedback clean
                         # tqdm.write(f"[DOC] {os.path.basename(path)}") 
                         ocr_text = self.ocr.extract_text(path)
                    
                    # 3. Metadata
                    metadata = self._extract_metadata(path)
                    
                    # 4. Save
                    try:
                        stat = os.stat(path)
                        self.db.add_photo(path, stat.st_size, stat.st_mtime, embedding, metadata, ocr_text)
                    except Exception as e:
                        print(f"DB Error {path}: {e}")
                        
            except Exception as e:
                print(f"Batch Error: {e}")

    def _extract_metadata(self, image_path: str) -> dict:
        """Extracts comprehensive metadata including GPS, device info, and camera settings."""
        meta = {}
        try:
            from PIL import Image, ExifTags
            from PIL.ExifTags import TAGS, GPSTAGS
            
            with Image.open(image_path) as img:
                # Basic image properties
                meta['width'], meta['height'] = img.size
                meta['format'] = img.format
                meta['mode'] = img.mode
                
                # Extract EXIF data
                exif = img._getexif()
                if exif:
                    # Device Information
                    for tag, value in exif.items():
                        tag_name = TAGS.get(tag, tag)
                        
                        # Camera/Device Details
                        if tag_name == 'Make':
                            meta['device_make'] = str(value).strip()
                        elif tag_name == 'Model':
                            meta['device_model'] = str(value).strip()
                        elif tag_name == 'Software':
                            meta['software'] = str(value).strip()
                        
                        # Date/Time Information
                        elif tag_name == 'DateTimeOriginal':
                            meta['date_taken'] = str(value)
                        elif tag_name == 'DateTime':
                            meta['date_modified'] = str(value)
                        
                        # Camera Settings
                        elif tag_name == 'ISOSpeedRatings':
                            meta['iso'] = int(value)
                        elif tag_name == 'FNumber':
                            meta['aperture'] = f"f/{float(value):.1f}"
                        elif tag_name == 'ExposureTime':
                            meta['shutter_speed'] = f"1/{int(1/float(value))}" if float(value) < 1 else f"{float(value)}s"
                        elif tag_name == 'FocalLength':
                            meta['focal_length'] = f"{float(value):.1f}mm"
                        elif tag_name == 'Flash':
                            meta['flash'] = 'Yes' if value & 1 else 'No'
                        
                        # Image Orientation
                        elif tag_name == 'Orientation':
                            meta['orientation'] = int(value)
                        
                        # GPS Information
                        elif tag_name == 'GPSInfo':
                            gps_data = {}
                            for gps_tag in value:
                                gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                                gps_data[gps_tag_name] = value[gps_tag]
                            
                            # Extract GPS coordinates
                            if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                                lat = self._convert_gps_to_decimal(
                                    gps_data['GPSLatitude'],
                                    gps_data.get('GPSLatitudeRef', 'N')
                                )
                                lon = self._convert_gps_to_decimal(
                                    gps_data['GPSLongitude'],
                                    gps_data.get('GPSLongitudeRef', 'E')
                                )
                                meta['gps_latitude'] = lat
                                meta['gps_longitude'] = lon
                                meta['location'] = f"{lat:.6f}, {lon:.6f}"
                                
                                # Resolve location name offline
                                location_name = self._resolve_location_name(lat, lon)
                                if location_name:
                                    meta['location_name'] = location_name
                            
                            # Altitude
                            if 'GPSAltitude' in gps_data:
                                altitude = float(gps_data['GPSAltitude'])
                                meta['gps_altitude'] = f"{altitude:.1f}m"
                            
                            # GPS Timestamp
                            if 'GPSDateStamp' in gps_data and 'GPSTimeStamp' in gps_data:
                                gps_date = gps_data['GPSDateStamp']
                                gps_time = gps_data['GPSTimeStamp']
                                meta['gps_timestamp'] = f"{gps_date} {gps_time[0]}:{gps_time[1]}:{gps_time[2]}"
                
        except Exception as e:
            # Silently fail but log if needed
            pass
        
        return meta
    
    def _convert_gps_to_decimal(self, coord, ref):
        """Convert GPS coordinates from degrees/minutes/seconds to decimal format."""
        try:
            degrees = float(coord[0])
            minutes = float(coord[1])
            seconds = float(coord[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        except:
            return 0.0
    
    def _resolve_location_name(self, lat, lon):
        """
        Convert GPS coordinates to human-readable location name (offline).
        Returns: "District, City, State, Country" or None if geocoder unavailable
        """
        if not GEOCODER_AVAILABLE:
            return None
        
        try:
            # Reverse geocode (completely offline, uses local database)
            results = rg.search((lat, lon), mode=1)  # mode=1 for single result
            
            if results and len(results) > 0:
                result = results[0]
                
                # Build detailed location string
                parts = []
                
                # Add district/admin2 (more specific than city)
                if result.get('admin2'):
                    parts.append(result['admin2'])
                
                # Add city/locality (if different from admin2)
                if result.get('name') and result.get('name') != result.get('admin2'):
                    parts.append(result['name'])
                
                # Add state/admin1
                if result.get('admin1'):
                    parts.append(result['admin1'])
                
                # Add country name (expand country code)
                if result.get('cc'):
                    country_map = {
                        'IN': 'India', 'US': 'United States', 'GB': 'United Kingdom',
                        'CN': 'China', 'JP': 'Japan', 'DE': 'Germany', 'FR': 'France',
                        'CA': 'Canada', 'AU': 'Australia', 'BR': 'Brazil'
                    }
                    country = country_map.get(result['cc'], result['cc'])
                    parts.append(country)
                
                return ', '.join(parts) if parts else None
        except Exception:
            return None
        
        return None

if __name__ == "__main__":
    # Test
    # Provide a directory path to scan
    import sys
    if len(sys.argv) > 1:
        scan_dir = sys.argv[1]
    else:
        scan_dir = "." # Scan current dir by default
    
    scanner = PhotoScanner(os.path.join("assets", "mobileclip_s1_datacompdr_first.tflite"))
    scanner.scan_directory(scan_dir)
