import easyocr
import os
import torch
import traceback
import cv2
import numpy as np

class OCRHandler:
    def __init__(self, languages=['en']):
        """
        Initialize EasyOCR reader.
        Args:
            languages: List of language codes to support (default: ['en'])
        """
        # Check availability of GPU
        use_gpu = torch.cuda.is_available()
        print(f"Initializing OCR Engine (GPU={use_gpu})...")
        
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file.
        Returns:
            Combined string of all detected text.
        """
        if not os.path.exists(image_path):
            return ""
        
        try:
            # Load as grayscale numpy array to avoid EasyOCR's internal
            # reformat_input bug where img.shape unpacking fails on 3-channel images
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return ""
            
            results = self.reader.readtext(img, detail=0)
            # Ensure we handle both formats (strings or tuples)
            texts = []
            for item in results:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, (list, tuple)):
                    texts.append(str(item[1]) if len(item) > 1 else str(item[0]))
            return " ".join(texts)
        except Exception as e:
            print(f"OCR Failed for {image_path}: {e}")
            traceback.print_exc()
            return ""
