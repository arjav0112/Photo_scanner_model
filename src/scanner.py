
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
        """Recursively scans a directory for photos with incremental indexing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        scan_start = time.time()
        print(f"Scanning directory: {directory}")
        
        # ============================================================
        # INCREMENTAL SCANNING - Detect new, modified, and deleted files
        # ============================================================
        scanned_data = self.db.get_scanned_paths_with_mtime()  # {path: (mtime, size)}
        
        # Collect all files currently on disk
        disk_files = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    full_path = os.path.abspath(os.path.join(root, file))
                    try:
                        stat = os.stat(full_path)
                        disk_files[full_path] = (stat.st_mtime, stat.st_size)
                    except OSError:
                        continue
        
        # Categorize files
        new_files = []
        modified_files = []
        
        for path, (mtime, size) in disk_files.items():
            if path not in scanned_data:
                new_files.append(path)
            else:
                old_mtime, old_size = scanned_data[path]
                if abs(mtime - old_mtime) > 1.0 or size != old_size:
                    modified_files.append(path)
        
        # Detect deleted files
        deleted_files = [p for p in scanned_data if p not in disk_files]
        
        # Remove deleted files from DB
        if deleted_files:
            self.db.remove_photos(deleted_files)
        
        files_to_process = new_files + modified_files
        
        if not files_to_process and not deleted_files:
            print("No new or modified photos found.")
            return
        
        print(f"Scan Summary:")
        print(f"   New files: {len(new_files)}")
        print(f"   Modified files: {len(modified_files)}")
        print(f"   Deleted files: {len(deleted_files)}")
        print(f"   Total to process: {len(files_to_process)}")
        
        if not files_to_process:
            self._rebuild_faiss_index()
            return
        
        batch_size = 16
        modified_set = set(modified_files)
        
        def chunked(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        for batch_idx, batch_paths in enumerate(tqdm(chunked(files_to_process, batch_size), total=len(files_to_process)//batch_size + 1, desc="Scanning Batches")):
            try:
                if batch_idx % 5 == 0:
                     self.model.log_memory(f"Scanning Batch {batch_idx}")
                     
                # 1. Batch Embeddings (GPU-bound, stays serial)
                embeddings = self.model.get_image_embeddings_batch(batch_paths)
                
                if embeddings is None:
                    continue
                
                # 2. PARALLEL metadata + OCR extraction (I/O-bound)
                metadata_results = {}
                ocr_results = {}
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit metadata extraction tasks
                    meta_futures = {
                        executor.submit(self._extract_metadata, path): path 
                        for path in batch_paths
                    }
                    
                    # Submit OCR tasks for document-like images
                    ocr_futures = {}
                    for i, path in enumerate(batch_paths):
                        embedding = embeddings[i]
                        if np.all(embedding == 0):
                            continue
                        emb_norm = embedding / np.linalg.norm(embedding)
                        sim_score = np.dot(emb_norm, self.doc_embedding)
                        if sim_score > self.doc_threshold:
                            future = executor.submit(self.ocr.extract_text, path)
                            ocr_futures[future] = path
                    
                    # Collect metadata results
                    for future in as_completed(meta_futures):
                        path = meta_futures[future]
                        try:
                            metadata_results[path] = future.result()
                        except Exception:
                            metadata_results[path] = {}
                    
                    # Collect OCR results
                    for future in as_completed(ocr_futures):
                        path = ocr_futures[future]
                        try:
                            ocr_results[path] = future.result()
                        except Exception:
                            ocr_results[path] = ""
                
                # 3. Save to database
                for i, path in enumerate(batch_paths):
                    embedding = embeddings[i]
                    if np.all(embedding == 0):
                        continue
                    
                    metadata = metadata_results.get(path, {})
                    ocr_text = ocr_results.get(path, "")
                    
                    try:
                        stat = os.stat(path)
                        if path in modified_set:
                            self.db.update_photo(path, stat.st_size, stat.st_mtime, embedding, metadata, ocr_text)
                        else:
                            self.db.add_photo(path, stat.st_size, stat.st_mtime, embedding, metadata, ocr_text)
                    except Exception as e:
                        print(f"DB Error {path}: {e}")
                    
            except Exception as e:
                print(f"Batch Error: {e}")
        
        # Rebuild FAISS index after scan
        self._rebuild_faiss_index()
        
        elapsed = time.time() - scan_start
        print(f"\nScan completed in {elapsed:.1f}s")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index after scan changes."""
        try:
            from .faiss_index import FAISSIndex
            faiss_idx = FAISSIndex()
            ids, embeddings = self.db.get_all_embeddings_with_ids()
            if len(ids) > 0:
                faiss_idx.build_index(ids, embeddings)
                faiss_idx.save()
                print("FAISS search index rebuilt")
        except Exception as e:
            print(f"FAISS index rebuild failed: {e}")

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
                                
                                # Resolve location name offline (only if valid coordinates)
                                if lat != 0.0 and lon != 0.0 and abs(lat) <= 90 and abs(lon) <= 180:
                                    location_name = self._resolve_location_name(lat, lon)
                                    if location_name:
                                        meta['location_name'] = location_name
                            
                            # Altitude
                            if 'GPSAltitude' in gps_data:
                                altitude = float(gps_data['GPSAltitude'])
                                meta['gps_altitude'] = f"{altitude:.1f}m"
                                meta['altitude'] = altitude  # Add numeric altitude for scoring
                            
                            # GPS Timestamp
                            if 'GPSDateStamp' in gps_data and 'GPSTimeStamp' in gps_data:
                                gps_date = gps_data['GPSDateStamp']
                                gps_time = gps_data['GPSTimeStamp']
                                meta['gps_timestamp'] = f"{gps_date} {gps_time[0]}:{gps_time[1]}:{gps_time[2]}"
            
                # Create combined 'device' field for metadata scorer compatibility
                if 'device_make' in meta or 'device_model' in meta:
                    make = meta.get('device_make', '')
                    model = meta.get('device_model', '')
                    if make and model:
                        meta['device'] = f"{make} {model}"
                    elif make:
                        meta['device'] = make
                    elif model:
                        meta['device'] = model
            
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
        
        Only called when valid GPS coordinates exist.
        """
        # Early return if geocoder not available
        if not GEOCODER_AVAILABLE:
            return None
        
        # Validate coordinates
        if lat == 0.0 and lon == 0.0:
            return None
        if abs(lat) > 90 or abs(lon) > 180:
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
