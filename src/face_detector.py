
import os
import numpy as np
import cv2
from typing import List, Optional


# InsightFace model is downloaded on first use to ~/.insightface/models/
# Model used: buffalo_l (RetinaFace detection + ArcFace R100 recognition)
_face_app = None
_FACE_MODEL = "buffalo_l"


def get_face_app():
    """Lazily initialise the InsightFace app (downloads model on first run)."""
    global _face_app
    if _face_app is None:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=_FACE_MODEL,
            providers=["CPUExecutionProvider"],  # swap to CUDAExecutionProvider if GPU
        )
        # det_size: input resolution for the detector – 640 is a good balance
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
    return _face_app


def load_image_for_insightface(image_path: str) -> Optional[np.ndarray]:
    """Load image as BGR numpy array (InsightFace expects BGR)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            # fallback via PIL for exotic formats
            from PIL import Image
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


def detect_faces(image_path: str) -> List[dict]:
    """
    Detect all faces in an image and return their embeddings + metadata.

    Returns a list of dicts, one per detected face:
        {
            "bbox": [x1, y1, x2, y2],   # float pixels
            "det_score": float,           # detection confidence [0-1]
            "embedding": np.ndarray,      # 512-d ArcFace L2-normalised vector
            "age":   int or None,
            "gender": "M" | "F" | None,
        }
    Only faces with det_score >= 0.5 and a valid embedding are returned.
    """
    img = load_image_for_insightface(image_path)
    if img is None:
        return []

    try:
        app = get_face_app()
        faces = app.get(img)
    except Exception as e:
        print(f"[FaceDetector] InsightFace error on {image_path}: {e}")
        return []

    results = []
    for face in faces:
        det_score = float(face.det_score) if hasattr(face, "det_score") else 0.0
        if det_score < 0.5:
            continue

        emb = face.embedding  # already L2-normalised by InsightFace
        if emb is None or len(emb) == 0:
            continue

        # Ensure it's a proper float32 unit vector
        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            continue
        emb = emb / norm

        bbox = face.bbox.tolist() if hasattr(face, "bbox") else [0, 0, 0, 0]

        age = None
        gender = None
        if hasattr(face, "age") and face.age is not None:
            try:
                age = int(face.age)
            except Exception:
                pass
        if hasattr(face, "gender") and face.gender is not None:
            gender = "M" if int(face.gender) == 1 else "F"

        results.append(
            {
                "bbox": bbox,
                "det_score": det_score,
                "embedding": emb,
                "age": age,
                "gender": gender,
            }
        )

    return results
