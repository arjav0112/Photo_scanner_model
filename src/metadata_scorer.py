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
