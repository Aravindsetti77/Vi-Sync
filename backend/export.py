import argparse
import os
import sys

def convert_to_onnx(model_id_or_path: str, output_dir: str):
    """
    Converts a standard PyTorch / HuggingFace model to ONNX format.
    The output directory will contain model.onnx, config.json, and tokenizer files,
    which is the exact format FastEmbed requires for custom local models.
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: Missing required packages.", file=sys.stderr)
        print("Please install them by running:", file=sys.stderr)
        print("pip install optimum[onnxruntime] transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model '{model_id_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    
    # Export the model to ONNX
    print("Exporting PyTorch model to ONNX. This may take a moment...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_id_or_path, export=True)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving ONNX format and tokenizer to '{output_dir}'...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    print("\n✅ Conversion complete!")
    print(f"Your model is now located in '{output_dir}' and is ready to be loaded by FastEmbed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a local PyTorch model (or HuggingFace model) to ONNX format for FastEmbed.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local PyTorch model directory, or a HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="onnx_model", help="Directory to save the ONNX model (default: onnx_model)")
    args = parser.parse_args()
    
    convert_to_onnx(args.model_path, args.output_dir)
