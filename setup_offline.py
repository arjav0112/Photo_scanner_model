from sentence_transformers import SentenceTransformer
import os
import shutil

def setup_offline_model():
    model_name = "clip-ViT-B-32"
    local_path = os.path.join("assets", "local_clip_model")
    
    print(f" preparing offline model: {model_name}...")
    
    # 1. Load model (will use cache if available, or download if net exists)
    # We set local_files_only=False first to ensure we get it, 
    # but if offline, it relies on cache.
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"First attempt failed: {str(e)}")
        print("Trying to force load from cache...")
        # Try finding in cache by setting offline environment
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = SentenceTransformer(model_name)
        except Exception as e2:
            print(f"FATAL: Could not load model to save it. You need internet for the first run, or a populated cache.")
            print(f"Error: {e2}")
            return

    # 2. Save model to local assets folder
    if os.path.exists(local_path):
        print(f"Removing existing local model at {local_path}...")
        shutil.rmtree(local_path)
        
    print(f"Saving model to {local_path}...")
    model.save(local_path)
    print("Success! The model is now saved locally.")
    print(f"Location: {os.path.abspath(local_path)}")
    print("You can now run the scanner without internet.")

if __name__ == "__main__":
    setup_offline_model()
