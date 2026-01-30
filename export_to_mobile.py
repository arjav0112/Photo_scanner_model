import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer, AutoProcessor
from pathlib import Path

def export_model_for_mobile():
    print("Exporting CLIP model to ONNX (Mobile Optimized)...")
    
    model_id = "openai/clip-vit-base-patch32" # Matches sentence-transformers/clip-ViT-B-32 base
    output_dir = Path("assets/mobile_model_quantized")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print(f"Loading and Exporting {model_id}...")
    
    # 1. Export to ONNX
    # We export the model directly using Optimum which handles tracing
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Save base ONNX
    onnx_path = output_dir / "model.onnx"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"ONNX model saved to {output_dir}")
    print("To run this on mobile, use ONNX Runtime Mobile.")
    
    # 2. Quantization (INT8)
    print("Applying Quantization (INT8)...")
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    
    quantizer = ORTQuantizer.from_pretrained(model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    quantized_path = output_dir / "model_int8.onnx"
    
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=dqconfig,
    )
    
    print(f"Quantized model saved to {output_dir}")
    print("Optimization Complete.")

if __name__ == "__main__":
    export_model_for_mobile()
