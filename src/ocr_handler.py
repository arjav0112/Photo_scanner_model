import easyocr
import os
import torch

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
            # detail=0 returns just the text list, not coordinates
            results = self.reader.readtext(image_path, detail=0)
            return " ".join(results)
        except Exception as e:
            print(f"OCR Failed for {image_path}: {e}")
            return ""
