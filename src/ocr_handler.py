import os

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'

import paddle
paddle.set_flags({'FLAGS_use_mkldnn': False})
paddle.disable_signal_handler()

_orig_config_init = paddle.inference.Config.__init__
def _patched_config_init(self, *args, **kwargs):
    _orig_config_init(self, *args, **kwargs)
    self.disable_mkldnn()
paddle.inference.Config.__init__ = _patched_config_init

from paddleocr import PaddleOCR
import traceback
import cv2
import numpy as np

LANGUAGE_MAP = {
    'en': 'en',
    'hi': 'hi',
    'hindi': 'hi',
    'devanagari': 'devanagari',
    'mr': 'mr',
    'ne': 'ne',
    'ta': 'ta',
    'te': 'te',
    'ka': 'ka',
    'gu': 'gu',
    'pa': 'pa',
    'bn': 'bn',
    'ur': 'ur',
    'ch': 'ch',
    'japan': 'japan',
    'korean': 'korean',
    'ar': 'ar',
    'fr': 'french',
    'de': 'german',
}

OCR_MAX_SIDE = 640


class OCRHandler:
    def __init__(self, languages=None, min_confidence=0.5):
        """
        Initialize PaddleOCR reader with speed-optimized settings.
        Args:
            languages: List of language codes (default: ['en']).
            min_confidence: Minimum confidence threshold (0-1). Default: 0.5
        """
        if languages is None:
            languages = ['en']

        lang = LANGUAGE_MAP.get(languages[0], languages[0])
        print(f"Initializing PaddleOCR Engine (lang={lang})...")

        self.reader = PaddleOCR(
            ocr_version='PP-OCRv3',
            use_angle_cls=False,
            lang=lang,
            show_log=False,
            enable_mkldnn=False,
            det_limit_side_len=640,
            det_db_score_mode='fast',
            rec_batch_num=16,
        )
        self.min_confidence = min_confidence

    def _preprocess_for_ocr(self, image_path: str) -> np.ndarray:
        """
        Preprocess image specifically for OCR speed and accuracy.
        These transforms are applied ONLY for OCR — the original image
        is still used for YOLO, visual analysis, and embeddings.

        Pipeline:
          1. Load image
          2. Downscale if too large (biggest speed win)
          3. Grayscale conversion (3x less data, better for text)
          4. Normalize pixel values to 0-255 range
          5. Adaptive binarization (removes noise, isolates text)

        Returns:
            Preprocessed numpy array ready for OCR, or None on failure.
        """
        
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > OCR_MAX_SIDE:
            scale = OCR_MAX_SIDE / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mn, mx = gray.min(), gray.max()
        if mx > mn:
            gray = ((gray - mn) * (255.0 / (mx - mn))).astype(np.uint8)

        return gray

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using PaddleOCR with preprocessing.
        Returns:
            Combined string of all detected text (confidence-filtered).
        """

        if not os.path.exists(image_path):
            return ""

        try:
            processed = self._preprocess_for_ocr(image_path)
            if processed is None:
                return ""

            det_result = self.reader.ocr(processed, det=True, rec=False, cls=False)

            if not det_result or not det_result[0] or len(det_result[0]) == 0:
                return ""

            result = self.reader.ocr(processed, cls=False)

            if not result or not result[0]:
                return ""

            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence >= self.min_confidence and text.strip():
                        texts.append(text.strip())

            return " ".join(texts)
        except Exception as e:
            print(f"OCR Failed for {image_path}: {e}")
            traceback.print_exc()
            return ""
