"""
Enhanced Image Analysis Module
Extracts visual metadata: face detection, scene classification, color analysis, quality scoring.
Uses YOLOv8n for face detection, heuristics for scene/color/quality.
"""

import numpy as np
from PIL import Image
import os

_yolo_model = None

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


class ImageAnalyzer:
    """Extracts enhanced visual metadata from images."""
    
    COLOR_RANGES = {
        'red':      ((0, 70, 50), (10, 255, 255)),
        'red2':     ((170, 70, 50), (179, 255, 255)),  
        'orange':   ((11, 70, 50), (25, 255, 255)),
        'yellow':   ((26, 70, 50), (34, 255, 255)),
        'green':    ((35, 70, 50), (85, 255, 255)),
        'blue':     ((86, 70, 50), (130, 255, 255)),
        'purple':   ((131, 70, 50), (160, 255, 255)),
        'pink':     ((161, 50, 50), (169, 255, 255)),
        'white':    ((0, 0, 200), (179, 30, 255)),
        'black':    ((0, 0, 0), (179, 255, 50)),
        'gray':     ((0, 0, 51), (179, 30, 199)),
        'brown':    ((10, 50, 30), (30, 200, 150)),
    }
    
    SKY_BLUE_RANGE = ((90, 30, 150), (130, 255, 255))
    VEGETATION_RANGE = ((35, 40, 40), (85, 255, 255))
    
    def __init__(self):
        pass
    
    def analyze(self, image_path: str) -> dict:
        """
        Run all visual analyzers on an image.
        
        Returns dict with keys: face_count, has_faces, face_category,
        scene_type, scene_environment, dominant_colors, color_tone,
        color_vibrance, blur_score, is_blurry, exposure, quality_rating
        """
        meta = {}
        try:
            img = Image.open(image_path)
            img.load()
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            
            meta.update(self._detect_faces(image_path))
            meta.update(self._classify_scene(img_array))
            meta.update(self._analyze_colors(img_array))
            meta.update(self._score_quality(img_array))
            
        except Exception as e:
            pass
        
        return meta
    
    def _detect_faces(self, image_path: str) -> dict:
        """
        Detect faces/people using YOLOv8n.
        Uses 'person' class detection as proxy for face presence.
        """
        result = {
            'face_count': 0,
            'has_faces': False,
            'face_category': 'no_faces'
        }
        
        try:
            model = _get_yolo_model()
            detections = model(image_path, verbose=False, conf=0.3)
            
            if detections and len(detections) > 0:
                boxes = detections[0].boxes
                person_count = 0
                
                for box in boxes:
                    if int(box.cls[0]) == 0:
                        person_count += 1
                
                result['face_count'] = person_count
                result['has_faces'] = person_count > 0
                
                if person_count == 0:
                    result['face_category'] = 'no_faces'
                elif person_count == 1:
                    result['face_category'] = 'portrait'
                elif person_count == 2:
                    result['face_category'] = 'duo'
                else:
                    result['face_category'] = 'group'
        except Exception:
            pass
        
        return result
    
    def _classify_scene(self, img_array: np.ndarray) -> dict:
        """
        Classify scene as indoor/outdoor and natural/urban using color heuristics.
        
        Strategy:
        - Top 1/3 of image: check for sky-blue → outdoor
        - Overall: check for green vegetation → natural
        - High edge density + low green → urban
        - Low sky + low green → indoor
        """
        result = {
            'scene_type': 'unknown',
            'scene_environment': 'unknown'
        }
        
        try:
            import cv2
            
            h, w = img_array.shape[:2]
            scale = 256 / max(h, w)
            small = cv2.resize(img_array, (int(w * scale), int(h * scale)))
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            
            sh, sw = small.shape[:2]
            total_pixels = sh * sw
            
            top_third = hsv[:sh // 3, :, :]
            top_pixels = top_third.shape[0] * top_third.shape[1]
            
            sky_lower = np.array(self.SKY_BLUE_RANGE[0])
            sky_upper = np.array(self.SKY_BLUE_RANGE[1])
            sky_mask = cv2.inRange(top_third, sky_lower, sky_upper)
            sky_ratio = np.count_nonzero(sky_mask) / max(top_pixels, 1)
            
            veg_lower = np.array(self.VEGETATION_RANGE[0])
            veg_upper = np.array(self.VEGETATION_RANGE[1])
            veg_mask = cv2.inRange(hsv, veg_lower, veg_upper)
            green_ratio = np.count_nonzero(veg_mask) / max(total_pixels, 1)
            
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / max(total_pixels, 1)
            
            avg_brightness = np.mean(hsv[:, :, 2])
            
            if sky_ratio > 0.10:
                result['scene_type'] = 'outdoor'
            elif green_ratio > 0.15:
                result['scene_type'] = 'outdoor'
            elif avg_brightness > 160 and (sky_ratio > 0.05 or green_ratio > 0.08):
                result['scene_type'] = 'outdoor'
            else:
                result['scene_type'] = 'indoor'
            
            if result['scene_type'] == 'outdoor':
                if green_ratio > 0.15 and edge_ratio < 0.12:
                    result['scene_environment'] = 'natural'
                elif edge_ratio > 0.15:
                    result['scene_environment'] = 'urban'
                elif sky_ratio > 0.20 and green_ratio < 0.05:
                    result['scene_environment'] = 'sky'
                else:
                    result['scene_environment'] = 'mixed'
            else:
                result['scene_environment'] = 'indoor'
            
        except Exception:
            pass
        
        return result
    
    def _analyze_colors(self, img_array: np.ndarray) -> dict:
        """
        Analyze dominant colors, tone (warm/cool), and vibrance.
        
        Uses color histogram in HSV space for speed (no k-means).
        """
        result = {
            'dominant_colors': [],
            'color_tone': 'neutral',
            'color_vibrance': 'moderate'
        }
        
        try:
            import cv2
            
            h, w = img_array.shape[:2]
            scale = 128 / max(h, w)
            small = cv2.resize(img_array, (int(w * scale), int(h * scale)))
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            
            total_pixels = small.shape[0] * small.shape[1]
            
            color_counts = {}
            for color_name, (lower, upper) in self.COLOR_RANGES.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                count = np.count_nonzero(mask)
                if color_name == 'red2':
                    color_counts['red'] = color_counts.get('red', 0) + count
                else:
                    color_counts[color_name] = color_counts.get(color_name, 0) + count
            
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            dominant = [name for name, count in sorted_colors 
                       if count / max(total_pixels, 1) > 0.05][:3]
            result['dominant_colors'] = dominant if dominant else ['mixed']
            
            warm_colors = {'red', 'orange', 'yellow', 'brown'}
            cool_colors = {'blue', 'green', 'purple'}
            
            warm_pixels = sum(color_counts.get(c, 0) for c in warm_colors)
            cool_pixels = sum(color_counts.get(c, 0) for c in cool_colors)
            
            if warm_pixels > cool_pixels * 1.5:
                result['color_tone'] = 'warm'
            elif cool_pixels > warm_pixels * 1.5:
                result['color_tone'] = 'cool'
            else:
                result['color_tone'] = 'neutral'
            
            avg_saturation = np.mean(hsv[:, :, 1])
            if avg_saturation > 120:
                result['color_vibrance'] = 'vibrant'
            elif avg_saturation > 60:
                result['color_vibrance'] = 'moderate'
            else:
                result['color_vibrance'] = 'muted'
            
        except Exception:
            pass
        
        return result
    
    def _score_quality(self, img_array: np.ndarray) -> dict:
        """
        Score image quality: blur detection + exposure analysis.
        
        Blur: Laplacian variance (low = blurry)
        Exposure: Histogram analysis (clipped highlights/shadows)
        """
        result = {
            'blur_score': 0.0,
            'is_blurry': False,
            'exposure': 'unknown',
            'quality_rating': 'unknown'
        }
        
        try:
            import cv2
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            h, w = gray.shape
            scale = 512 / max(h, w)
            gray_resized = cv2.resize(gray, (int(w * scale), int(h * scale)))
            
            laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
            blur_score = float(laplacian.var())
            result['blur_score'] = round(blur_score, 1)
            result['is_blurry'] = blur_score < 100  
            
            hist = cv2.calcHist([gray_resized], [0], None, [256], [0, 256])
            hist = hist.flatten()
            total_px = gray_resized.shape[0] * gray_resized.shape[1]
            
            dark_ratio = np.sum(hist[:20]) / total_px
            bright_ratio = np.sum(hist[235:]) / total_px
            
            mean_brightness = np.mean(gray_resized)
            
            if dark_ratio > 0.40:
                result['exposure'] = 'underexposed'
            elif bright_ratio > 0.40:
                result['exposure'] = 'overexposed'
            elif mean_brightness < 60:
                result['exposure'] = 'dark'
            elif mean_brightness > 200:
                result['exposure'] = 'bright'
            else:
                result['exposure'] = 'well_exposed'
            
            quality_points = 0
            if not result['is_blurry']:
                quality_points += 2
            if blur_score > 300:
                quality_points += 1
            if result['exposure'] == 'well_exposed':
                quality_points += 2
            elif result['exposure'] in ('dark', 'bright'):
                quality_points += 1
            
            if quality_points >= 4:
                result['quality_rating'] = 'excellent'
            elif quality_points >= 3:
                result['quality_rating'] = 'good'
            elif quality_points >= 2:
                result['quality_rating'] = 'fair'
            else:
                result['quality_rating'] = 'poor'
            
        except Exception:
            pass
        
        return result
