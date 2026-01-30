# import tensorflow as tf
# import numpy as np
# import os

# model_path = os.path.join("assets", "mobileclip_s1_datacompdr_first.tflite")

# def inspect_model():
#     if not os.path.exists(model_path):
#         print(f"Error: Model not found at {model_path}")
#         return

#     with open("model_details.log", "w", encoding="utf-8") as f:
#         f.write(f"Loading model from {model_path}...\n")
#         try:
#             interpreter = tf.lite.Interpreter(model_path=model_path)
#             interpreter.allocate_tensors()
#         except Exception as e:
#             f.write(f"Failed to load model: {e}\n")
#             return

#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
        
#         f.write("\n" + "="*30 + "\n")
#         f.write("MODEL SIGNATURE INSPECTION\n")
#         f.write("="*30 + "\n")

#         f.write(f"\nNumber of Inputs: {len(input_details)}\n")
#         for i, detail in enumerate(input_details):
#             f.write(f"\nInput {i}:\n")
#             f.write(f"  Name: {detail['name']}\n")
#             f.write(f"  Shape: {detail['shape']}\n")
#             f.write(f"  Type: {detail['dtype']}\n")
#             f.write(f"  Index: {detail['index']}\n")

#         f.write(f"\nNumber of Outputs: {len(output_details)}\n")
#         for i, detail in enumerate(output_details):
#             f.write(f"\nOutput {i}:\n")
#             f.write(f"  Name: {detail['name']}\n")
#             f.write(f"  Shape: {detail['shape']}\n")
#             f.write(f"  Type: {detail['dtype']}\n")
#             f.write(f"  Index: {detail['index']}\n")
        
#         f.write("\n" + "="*30 + "\n")
        
#         # Check for signatures
#         signatures = interpreter.get_signature_list()
#         if signatures:
#             f.write("\nSignatures found:\n")
#             for key, value in signatures.items():
#                 f.write(f"  {key}: {value}\n")
#         else:
#             f.write("\nNo named signatures found (using default).\n")

# if __name__ == "__main__":
#     inspect_model()
