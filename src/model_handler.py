from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

import psutil

class MobileCLIPHandler:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initializes the SentenceTransformer CLIP model.
        Args:
           model_name: The name of the model to load from HuggingFace.
                       Defaults to 'clip-ViT-B-32' (Strong performance, reasonable speed).
        """
        self.model_name = model_name
        token = os.getenv("HF_TOKEN")
        
        self.log_memory("Before Model Load")
        
        local_path = os.path.join("assets", "local_clip_model")
        if os.path.exists(local_path):
            print(f"Loading Offline Model from: {local_path}...")
            load_target = local_path
            load_token = None
        else:
            print(f"Loading SentenceTransformer CLIP model: {model_name} (Online/Cache)...")
            load_target = model_name
            load_token = token
            if token:
               print("  Using HF_TOKEN for authentication.")
            
        try:
            self.model = SentenceTransformer(load_target, token=load_token)
            print("Model loaded successfully.")
            self.log_memory("After Model Load")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

    def log_memory(self, stage: str):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mb = mem_info.rss / 1024 / 1024
        print(f"[{stage}] Memory Usage: {mb:.2f} MB")

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generates an embedding for a single image.
        """
        res = self.get_image_embeddings_batch([image_path])
        return res[0] if res is not None else None

    def get_image_embeddings_batch(self, image_paths: list) -> np.ndarray:
        """
        Generates embeddings for a batch of images efficiently.
        Returns numpy array of shape (N, 512) or None if all fail.
        """
        valid_images = []
        valid_indices = []
        
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    img.load() 
                    valid_images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        if not valid_images:
            return None
            
        try:
            embeddings = self.model.encode(valid_images, batch_size=len(valid_images), show_progress_bar=False)
            
            dim = embeddings.shape[1]
            result = np.zeros((len(image_paths), dim), dtype=np.float32)
            
            for i, idx in enumerate(valid_indices):
                result[idx] = embeddings[i]
                
            return result
            
        except Exception as e:
            print(f"Batch encoding failed: {e}")
            return None

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding for the given text using SentenceTransformer.
        """
        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return np.zeros(512)

    def identify_indices(self):
        pass

if __name__ == "__main__":
    handler = MobileCLIPHandler()
    
    txt_emb = handler.get_text_embedding("dog")
    print(f"Text Embedding Shape: {txt_emb.shape}")
    print(f"Text Embedding First 5: {txt_emb[:5]}")
    
    test_img = "image/dog.png" 
    test_img = "image/dog.png" 
    if os.path.exists(test_img):
        img_emb = handler.get_image_embedding(test_img)
        print(f"Image Embedding Shape: {img_emb.shape}")
        
        sim = np.dot(img_emb / np.linalg.norm(img_emb), 
                    txt_emb / np.linalg.norm(txt_emb))
        print(f"Similarity (dog vs 'dog'): {sim:.4f}")
