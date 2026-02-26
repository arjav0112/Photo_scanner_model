"""
Metadata Scoring Module
Intelligently scores images based on metadata matching with search queries
"""

import re
from datetime import datetime
from typing import Dict, List, Tuple
import json


class MetadataScorer:
    """
    Scores images based on metadata relevance to search query.
    Handles location, date/time, device, camera settings, and more.
    """
    
    # Month name mappings
    MONTHS = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    def __init__(self):
        self.query_lower = ""
        self.query_tokens = []
        
    def set_query(self, query: str):
        """Set the search query and prepare for scoring"""
        self.query_lower = query.lower()
        self.query_tokens = self.query_lower.split()
    
    def score_metadata(self, metadata: Dict) -> Tuple[float, List[str]]:
        """
        Score a single image's metadata against the query.
        
        Returns:
            (score, match_reasons): Score and list of why it matched
        """
        if not metadata:
            return 0.0, []
        
        total_score = 0.0
        match_reasons = []
        
        # 1. Location-based scoring
        location_score, location_reasons = self._score_location(metadata)
        total_score += location_score
        match_reasons.extend(location_reasons)
        
        # 2. Date/Time-based scoring
        date_score, date_reasons = self._score_datetime(metadata)
        total_score += date_score
        match_reasons.extend(date_reasons)
        
        # 3. Device-based scoring
        device_score, device_reasons = self._score_device(metadata)
        total_score += device_score
        match_reasons.extend(device_reasons)
        
        # 4. Camera settings scoring
        camera_score, camera_reasons = self._score_camera_settings(metadata)
        total_score += camera_score
        match_reasons.extend(camera_reasons)
        
        # 5. Altitude/Elevation scoring
        altitude_score, altitude_reasons = self._score_altitude(metadata)
        total_score += altitude_score
        match_reasons.extend(altitude_reasons)
        
        # 6. Image properties scoring
        props_score, props_reasons = self._score_image_properties(metadata)
        total_score += props_score
        match_reasons.extend(props_reasons)
        
        # 7. Face detection scoring
        face_score, face_reasons = self._score_faces(metadata)
        total_score += face_score
        match_reasons.extend(face_reasons)
        
        # 8. Scene classification scoring
        scene_score, scene_reasons = self._score_scene(metadata)
        total_score += scene_score
        match_reasons.extend(scene_reasons)
        
        # 9. Color analysis scoring
        color_score, color_reasons = self._score_color(metadata)
        total_score += color_score
        match_reasons.extend(color_reasons)
        
        # 10. Quality scoring
        quality_score, quality_reasons = self._score_quality(metadata)
        total_score += quality_score
        match_reasons.extend(quality_reasons)
        
        return total_score, match_reasons
    
    def _score_location(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on GPS location and location name"""
        score = 0.0
        reasons = []
        
        location_name = metadata.get('location_name', '')
        if location_name:
            location_lower = location_name.lower()
            
            # Check for location keywords in query
            location_keywords = ['delhi', 'india', 'pitampura', 'mumbai', 'bangalore', 
                                'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad',
                                'city', 'town', 'village', 'country', 'state']
            
            for keyword in location_keywords:
                if keyword in self.query_lower and keyword in location_lower:
                    score += 1.5  # Strong location match
                    reasons.append(f"Location: {location_name}")
                    break
            
            # Partial location match
            for token in self.query_tokens:
                if len(token) > 3 and token in location_lower:
                    score += 1.0
                    reasons.append(f"Location contains: {token}")
                    break
        
        # Check for GPS-related queries
        if metadata.get('gps_latitude') and metadata.get('gps_longitude'):
            if any(word in self.query_lower for word in ['gps', 'location', 'coordinates', 'map', 'place']):
                score += 0.5
                reasons.append("Has GPS data")
        
        return score, reasons
    
    def _score_datetime(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on date and time information"""
        score = 0.0
        reasons = []
        
        date_taken = metadata.get('date_taken', '')
        if not date_taken:
            return score, reasons
        
        try:
            # Parse date: "2026:02:01 14:45:43"
            date_obj = datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S")
            
            # Year matching
            year_str = str(date_obj.year)
            if year_str in self.query_lower:
                score += 1.5
                reasons.append(f"Year: {year_str}")
            
            # Month matching (by name or number)
            month_num = date_obj.month
            month_name = date_obj.strftime("%B").lower()  # "February"
            month_short = date_obj.strftime("%b").lower()  # "Feb"
            
            for token in self.query_tokens:
                if token in self.MONTHS and self.MONTHS[token] == month_num:
                    score += 1.5
                    reasons.append(f"Month: {month_name.capitalize()}")
                    break
            
            # Day matching
            day_str = str(date_obj.day)
            if day_str in self.query_tokens:
                score += 0.5
                reasons.append(f"Day: {day_str}")
            
            # Time-based keywords
            hour = date_obj.hour
            if 'morning' in self.query_lower and 5 <= hour < 12:
                score += 1.0
                reasons.append("Morning photo")
            elif 'afternoon' in self.query_lower and 12 <= hour < 17:
                score += 1.0
                reasons.append("Afternoon photo")
            elif 'evening' in self.query_lower and 17 <= hour < 21:
                score += 1.0
                reasons.append("Evening photo")
            elif 'night' in self.query_lower and (hour >= 21 or hour < 5):
                score += 1.0
                reasons.append("Night photo")
            
            # Recent/old keywords
            if 'recent' in self.query_lower or 'latest' in self.query_lower:
                days_ago = (datetime.now() - date_obj).days
                if days_ago < 7:
                    score += 1.5
                    reasons.append(f"Recent ({days_ago} days ago)")
                elif days_ago < 30:
                    score += 1.0
                    reasons.append(f"Recent ({days_ago} days ago)")
            
            if 'old' in self.query_lower or 'older' in self.query_lower:
                days_ago = (datetime.now() - date_obj).days
                if days_ago > 365:
                    score += 1.0
                    reasons.append(f"Old photo ({days_ago} days ago)")
        
        except (ValueError, TypeError):
            pass
        
        return score, reasons
    
    def _score_device(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on device/camera make and model"""
        score = 0.0
        reasons = []
        
        device = metadata.get('device', '')
        if device:
            device_lower = device.lower()
            
            # Brand matching
            brands = ['oneplus', 'samsung', 'apple', 'iphone', 'pixel', 'google', 
                     'xiaomi', 'oppo', 'vivo', 'realme', 'nokia', 'motorola',
                     'canon', 'nikon', 'sony', 'fujifilm', 'olympus']
            
            for brand in brands:
                if brand in self.query_lower and brand in device_lower:
                    score += 1.5
                    reasons.append(f"Device: {device}")
                    break
            
            # Generic device keywords
            if any(word in self.query_lower for word in ['phone', 'mobile', 'smartphone']):
                if any(word in device_lower for word in ['oneplus', 'samsung', 'iphone', 'pixel']):
                    score += 0.8
                    reasons.append("Phone photo")
            
            if any(word in self.query_lower for word in ['camera', 'dslr', 'professional']):
                if any(word in device_lower for word in ['canon', 'nikon', 'sony']):
                    score += 0.8
                    reasons.append("Camera photo")
        
        return score, reasons
    
    def _score_camera_settings(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on camera settings (ISO, aperture, etc.)"""
        score = 0.0
        reasons = []
        
        # ISO scoring
        iso = metadata.get('iso')
        if iso:
            if 'low light' in self.query_lower or 'night' in self.query_lower or 'dark' in self.query_lower:
                if iso >= 800:
                    score += 1.0
                    reasons.append(f"High ISO ({iso}) - Low light")
            
            if 'bright' in self.query_lower or 'sunny' in self.query_lower or 'daylight' in self.query_lower:
                if iso <= 200:
                    score += 0.8
                    reasons.append(f"Low ISO ({iso}) - Bright conditions")
        
        # Aperture scoring
        aperture = metadata.get('aperture')
        if aperture:
            try:
                f_num = float(aperture.replace('f/', ''))
                
                if 'portrait' in self.query_lower or 'bokeh' in self.query_lower or 'blur' in self.query_lower:
                    if f_num <= 2.8:
                        score += 1.0
                        reasons.append(f"Wide aperture ({aperture}) - Portrait/Bokeh")
                
                if 'landscape' in self.query_lower or 'sharp' in self.query_lower:
                    if f_num >= 8.0:
                        score += 0.8
                        reasons.append(f"Narrow aperture ({aperture}) - Landscape")
            except (ValueError, AttributeError):
                pass
        
        # Flash scoring
        flash = metadata.get('flash', '')
        if flash:
            if 'flash' in self.query_lower:
                if 'yes' in flash.lower() or 'fired' in flash.lower():
                    score += 1.0
                    reasons.append("Flash used")
            
            if 'no flash' in self.query_lower or 'natural light' in self.query_lower:
                if 'no' in flash.lower() or 'not' in flash.lower():
                    score += 0.8
                    reasons.append("No flash")
        
        return score, reasons
    
    def _score_altitude(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on altitude/elevation"""
        score = 0.0
        reasons = []
        
        altitude = metadata.get('altitude')
        if altitude:
            try:
                alt_meters = float(altitude)
                
                # Mountain/hill keywords
                if any(word in self.query_lower for word in ['mountain', 'hill', 'peak', 'summit', 'trek', 'hiking']):
                    if alt_meters > 1000:
                        score += 1.5
                        reasons.append(f"High altitude ({alt_meters:.0f}m) - Mountain")
                    elif alt_meters > 500:
                        score += 1.0
                        reasons.append(f"Elevated ({alt_meters:.0f}m) - Hill")
                
                # Sea level / beach keywords
                if any(word in self.query_lower for word in ['beach', 'sea', 'ocean', 'coast', 'shore']):
                    if alt_meters < 50:
                        score += 1.0
                        reasons.append(f"Sea level ({alt_meters:.0f}m) - Beach/Coast")
            
            except (ValueError, TypeError):
                pass
        
        return score, reasons
    
    def _score_image_properties(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on image dimensions, format, etc."""
        score = 0.0
        reasons = []
        
        width = metadata.get('width')
        height = metadata.get('height')
        
        if width and height:
            try:
                w = int(width)
                h = int(height)
                megapixels = (w * h) / 1_000_000
                
                # Resolution keywords
                if any(word in self.query_lower for word in ['high resolution', 'hd', 'quality', 'large']):
                    if megapixels > 8:
                        score += 0.8
                        reasons.append(f"High resolution ({megapixels:.1f}MP)")
                
                if any(word in self.query_lower for word in ['low resolution', 'small', 'thumbnail']):
                    if megapixels < 1:
                        score += 0.5
                        reasons.append(f"Low resolution ({megapixels:.1f}MP)")
                
                # Orientation keywords
                if 'vertical' in self.query_lower or 'portrait' in self.query_lower:
                    if h > w:
                        score += 0.5
                        reasons.append("Vertical orientation")
                
                if 'horizontal' in self.query_lower or 'landscape' in self.query_lower:
                    if w > h:
                        score += 0.5
                        reasons.append("Horizontal orientation")
                
                if 'square' in self.query_lower:
                    if abs(w - h) < min(w, h) * 0.1:  # Within 10% of square
                        score += 0.5
                        reasons.append("Square format")
            
            except (ValueError, TypeError):
                pass
        
        return score, reasons
    
    def _score_faces(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on face/person detection metadata"""
        score = 0.0
        reasons = []
        
        face_count = metadata.get('face_count', 0)
        face_category = metadata.get('face_category', 'no_faces')
        has_faces = metadata.get('has_faces', False)
        
        # "photos with faces" / "people" queries
        if any(w in self.query_lower for w in ['face', 'faces', 'people', 'person', 'someone']):
            if has_faces:
                score += 1.5
                reasons.append(f"Has {face_count} person(s) detected")
            else:
                score -= 0.3  # Slight penalty for no faces when user wants faces
        
        # "selfie" / "portrait" queries
        if any(w in self.query_lower for w in ['selfie', 'portrait']):
            if face_category == 'portrait':
                score += 1.5
                reasons.append("Single person (portrait/selfie)")
            elif face_category == 'duo':
                score += 0.5
        
        # "group photo" / "group" queries
        if any(w in self.query_lower for w in ['group', 'crowd', 'team', 'family']):
            if face_category == 'group':
                score += 1.5
                reasons.append(f"Group photo ({face_count} people)")
            elif face_category == 'duo':
                score += 0.8
                reasons.append("Duo photo (2 people)")
        
        # "no people" / "empty" queries
        if any(w in self.query_lower for w in ['no people', 'empty', 'nobody', 'no person']):
            if not has_faces:
                score += 1.0
                reasons.append("No people detected")
        
        return score, reasons
    
    def _score_scene(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on scene classification metadata"""
        score = 0.0
        reasons = []
        
        scene_type = metadata.get('scene_type', 'unknown')
        scene_env = metadata.get('scene_environment', 'unknown')
        
        if scene_type == 'unknown':
            return score, reasons
        
        # Indoor/outdoor queries
        if any(w in self.query_lower for w in ['outdoor', 'outside', 'open air']):
            if scene_type == 'outdoor':
                score += 1.5
                reasons.append(f"Outdoor scene ({scene_env})")
        
        if any(w in self.query_lower for w in ['indoor', 'inside', 'room', 'interior']):
            if scene_type == 'indoor':
                score += 1.5
                reasons.append("Indoor scene")
        
        # Nature queries
        if any(w in self.query_lower for w in ['nature', 'natural', 'green', 'vegetation', 'forest', 'park', 'garden']):
            if scene_env == 'natural':
                score += 1.5
                reasons.append("Natural environment")
        
        # Urban/city queries
        if any(w in self.query_lower for w in ['urban', 'city', 'street', 'building', 'architecture']):
            if scene_env == 'urban':
                score += 1.5
                reasons.append("Urban environment")
        
        # Sky queries
        if any(w in self.query_lower for w in ['sky', 'clouds', 'sunset', 'sunrise']):
            if scene_env == 'sky' or scene_type == 'outdoor':
                score += 1.0
                reasons.append("Sky visible")
        
        return score, reasons
    
    def _score_color(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on color analysis metadata"""
        score = 0.0
        reasons = []
        
        dominant_colors = metadata.get('dominant_colors', [])
        color_tone = metadata.get('color_tone', 'neutral')
        color_vibrance = metadata.get('color_vibrance', 'moderate')
        
        if not dominant_colors:
            return score, reasons
        
        # Warm/cool tone queries
        if any(w in self.query_lower for w in ['warm tone', 'warm color', 'warm photo', 'warm']):
            if color_tone == 'warm':
                score += 1.5
                reasons.append("Warm color tones")
        
        if any(w in self.query_lower for w in ['cool tone', 'cool color', 'cool photo', 'cool']):
            if color_tone == 'cool':
                score += 1.5
                reasons.append("Cool color tones")
        
        # Vibrance queries
        if any(w in self.query_lower for w in ['vibrant', 'colorful', 'saturated', 'vivid', 'bright color']):
            if color_vibrance == 'vibrant':
                score += 1.5
                reasons.append("Vibrant colors")
        
        if any(w in self.query_lower for w in ['muted', 'subtle', 'desaturated', 'faded', 'pastel']):
            if color_vibrance == 'muted':
                score += 1.5
                reasons.append("Muted colors")
        
        # Specific color queries
        color_keywords = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white', 'black', 'brown']
        for color in color_keywords:
            if color in self.query_lower and color in dominant_colors:
                score += 1.2
                reasons.append(f"Dominant color: {color}")
                break  # Only match one specific color
        
        return score, reasons
    
    def _score_quality(self, metadata: Dict) -> Tuple[float, List[str]]:
        """Score based on quality analysis metadata"""
        score = 0.0
        reasons = []
        
        quality_rating = metadata.get('quality_rating', 'unknown')
        is_blurry = metadata.get('is_blurry', False)
        exposure = metadata.get('exposure', 'unknown')
        blur_score = metadata.get('blur_score', 0)
        
        # Blur queries
        if any(w in self.query_lower for w in ['blurry', 'blur', 'out of focus', 'unfocused']):
            if is_blurry:
                score += 1.5
                reasons.append(f"Blurry photo (score: {blur_score})")
        
        if any(w in self.query_lower for w in ['sharp', 'clear', 'focused', 'crisp', 'in focus']):
            if not is_blurry and blur_score > 200:
                score += 1.5
                reasons.append(f"Sharp photo (score: {blur_score})")
        
        # Quality queries
        if any(w in self.query_lower for w in ['high quality', 'best quality', 'excellent', 'best photos', 'good quality']):
            if quality_rating in ('excellent', 'good'):
                score += 1.5
                reasons.append(f"Quality: {quality_rating}")
        
        if any(w in self.query_lower for w in ['low quality', 'bad quality', 'poor']):
            if quality_rating == 'poor':
                score += 1.0
                reasons.append("Low quality photo")
        
        # Exposure queries
        if any(w in self.query_lower for w in ['well exposed', 'good exposure', 'proper exposure']):
            if exposure == 'well_exposed':
                score += 1.0
                reasons.append("Well exposed")
        
        if any(w in self.query_lower for w in ['overexposed', 'too bright', 'blown out']):
            if exposure == 'overexposed':
                score += 1.0
                reasons.append("Overexposed")
        
        if any(w in self.query_lower for w in ['underexposed', 'too dark']):
            if exposure in ('underexposed', 'dark'):
                score += 1.0
                reasons.append("Underexposed/dark")
        
        return score, reasons


def score_batch_metadata(query: str, metadata_list: List[Dict]) -> Tuple[List[float], List[List[str]]]:
    """
    Score a batch of metadata against a query.
    
    Args:
        query: Search query string
        metadata_list: List of metadata dictionaries
    
    Returns:
        (scores, reasons_list): List of scores and list of match reasons for each image
    """
    scorer = MetadataScorer()
    scorer.set_query(query)
    
    scores = []
    reasons_list = []
    
    for metadata in metadata_list:
        score, reasons = scorer.score_metadata(metadata)
        scores.append(score)
        reasons_list.append(reasons)
    
    return scores, reasons_list
