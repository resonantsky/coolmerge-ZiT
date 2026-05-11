import json
import os
from safetensors import safe_open

# --- Configuration ---
# Your specific file path
MODEL_PATH = r"E:\SD.Next\models\Diffusers\MoodyRealMix-SDNQ-int8-svd-r32\transformer\diffusion_pytorch_model.safetensors"

def format_size(num_params):
    """Formats number of parameters into a readable string (M or B)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    return str(num_params)

def inspect_safetensors(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Create log filename based on the model filename
    log_filename = os.path.splitext(os.path.basename(file_path))[0] + ".log"
    output_buffer = []

    def log(message):
        """Helper to print to console and buffer for file writing."""
        print(message)
        output_buffer.append(message)

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            tensor_keys = f.keys()
            tensor_info = []
            total_params = 0
            
            for key in tensor_keys:
                shape = f.get_slice(key).get_shape()
                params = 1
                for dim in shape:
                    params *= dim
                total_params += params
                
                # Get dtype without loading data
                dtype = f.get_slice(key)[:0].dtype 
                tensor_info.append((key, shape, dtype))

            # --- Construct Output ---
            log("="*90)
            log(f"FILE: {os.path.basename(file_path)}")
            log(f"PATH: {file_path}")
            log(f"SIZE: {os.path.getsize(file_path) / (1024**3):.2f} GB")
            log(f"TOTAL PARAMS: {format_size(total_params)}")
            log("="*90)

            log("\n[ HEADER METADATA ]")
            if metadata:
                log(json.dumps(metadata, indent=4))
            else:
                log("No header metadata present.")

            log(f"\n[ TENSOR MAP ] ({len(tensor_keys)} tensors)")
            # Adjusted spacing for long Diffusion/Transformer keys
            log(f"{'Tensor Name':<70} | {'Shape':<20} | {'Dtype'}")
            log("-" * 110)
            
            for name, shape, dtype in sorted(tensor_info):
                shape_str = str(list(shape))
                log(f"{name[:68]:<70} | {shape_str:<20} | {dtype}")
                
            log("="*90)

            # --- Write to Log File ---
            with open(log_filename, "w", encoding="utf-8") as lf:
                lf.write("\n".join(output_buffer))
            
            print(f"\n[INFO] Data successfully written to: {log_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_safetensors(MODEL_PATH)