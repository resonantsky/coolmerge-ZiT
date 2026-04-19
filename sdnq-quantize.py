import os, torch
from colorama import Fore, Style, init
from diffusers import DiffusionPipeline

# This script is used to bake a quant into a model so that it does not have to be done live.

# Initialize colorama
init(autoreset=True)

# =============================================================================
# CONFIG — edit these before running
# =============================================================================

# Path to the source Diffusers pipeline folder (full pipeline directory)
SOURCE_MODEL    = "E:/SD.Next/models/Diffusers/"

# Parent folder where the quantized model will be saved
OUTPUT_DIR      = "E:/SD.Next/models/Diffusers"

# dtype to load the pipeline in before quantizing
LOAD_DTYPE      = torch.bfloat16          # torch.float16 or torch.bfloat16

# Which device to quantize on ("cuda", "cpu", "mps", "xpu")
QUANTIZATION_DEVICE = "cuda"
# Which device to keep the result on after quantization
RETURN_DEVICE       = "cuda"

# --- Quantization settings (map to SDNQConfig / sdnq_post_load_quant kwargs) ---
QUANT_DTYPE     = "uint4"   # int8, uint4, int4, uint2, fp8, fp16
                             # see sdnq.common.accepted_weight_dtypes for full list
GROUP_SIZE      = 0          # 0 = auto, -1 = disabled
USE_SVD         = True
SVD_RANK        = 32
SVD_STEPS       = 8
USE_DYNAMIC     = False      # use_dynamic_quantization: adds -dynamic to output name
QUANT_CONV      = False      # also quantize conv layers
QUANT_EMBEDDING = False      # also quantize embedding layers
DEQUANTIZE_FP32 = False      # keep scales in activation dtype to avoid Float/Half matmul mismatch at inference

# Modules that should NOT be quantized (keeps them at original dtype)
MODULES_TO_NOT_CONVERT = [
    "correction_coefs", "prediction_coefs",
    "lm_head", "embedding_projection",
]

# Which pipeline component to quantize.  "auto" = detect transformer then unet.
# Can also be "transformer" or "unet" to force a specific component.
TARGET_COMPONENT = "auto"

# Set to True to also quantize text encoder(s) (text_encoder, text_encoder_2, text_encoder_3).
# Uses a lighter int8 dtype by default since text encoders are smaller and more sensitive.
QUANT_TEXT_ENCODER       = True
TEXT_ENCODER_QUANT_DTYPE = "int8"   # typically int8 is safe for text encoders

# =============================================================================
# Console helpers (same style as coolmerge-ZiT)
# =============================================================================

def info(msg):
    print(f"\n{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")

def success(msg):
    print(f"\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {msg}")

def warn(msg):
    print(f"\n{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")

def error(msg):
    print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

# =============================================================================
# Naming helper
# =============================================================================

def build_output_name(source_path: str) -> str:
    """
    Derive output folder name from source model name and quant settings.
    Convention: {ModelName}-SDNQ-{dtype}[-svd-r{rank}][-dynamic]

    Examples matching existing models in models/Diffusers:
      FLUX.1-dev   + uint4 + svd r32  → FLUX.1-dev-SDNQ-uint4-svd-r32
      DivingZImageTurbo + int8        → DivingZImageTurbo-SDNQ-int8
    """
    model_name = os.path.basename(source_path.rstrip("/\\"))
    suffix = f"-SDNQ-{QUANT_DTYPE}"
    if USE_SVD:
        suffix += f"-svd-r{SVD_RANK}"
    if USE_DYNAMIC:
        suffix += "-dynamic"
    return model_name + suffix

# =============================================================================
# Main workflow
# =============================================================================

# Derive output path
output_name = build_output_name(SOURCE_MODEL)
output_path = os.path.join(OUTPUT_DIR, output_name)

info(f"Source model : {SOURCE_MODEL}")
info(f"Output path  : {output_path}")
info(f"Quant dtype  : {QUANT_DTYPE}  |  SVD={USE_SVD} rank={SVD_RANK}  |  dynamic={USE_DYNAMIC}")

# Guard: don't clobber an existing directory
if os.path.exists(output_path):
    error(f"Output directory already exists: {output_path}")
    error("Remove or rename it before running this script to avoid overwriting a previous result.")
    raise SystemExit(1)

# Import sdnq after the guard so a missing install is caught cleanly
try:
    from sdnq import SDNQConfig  # noqa: F401  registers sdnq into diffusers/transformers
    from sdnq import sdnq_post_load_quant
except ImportError as exc:
    error(f"sdnq is not installed: {exc}")
    error("Install it with:  pip install sdnq")
    raise SystemExit(1)

# --- Load full pipeline ---
info(f"Loading pipeline from {SOURCE_MODEL} (dtype={LOAD_DTYPE}) ...")
pipe = DiffusionPipeline.from_pretrained(SOURCE_MODEL, torch_dtype=LOAD_DTYPE)
success("Pipeline loaded.")

# --- Detect which component to quantize ---
component = None
component_name = None

if TARGET_COMPONENT == "auto":
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        component = pipe.transformer
        component_name = "transformer"
    elif hasattr(pipe, "unet") and pipe.unet is not None:
        component = pipe.unet
        component_name = "unet"
    else:
        error("Could not find a 'transformer' or 'unet' attribute on the loaded pipeline.")
        error("Set TARGET_COMPONENT = 'transformer' or 'unet' explicitly in the config.")
        raise SystemExit(1)
elif TARGET_COMPONENT in ("transformer", "unet"):
    component_name = TARGET_COMPONENT
    component = getattr(pipe, TARGET_COMPONENT, None)
    if component is None:
        error(f"Pipeline has no attribute '{TARGET_COMPONENT}'.")
        raise SystemExit(1)
else:
    error(f"Unknown TARGET_COMPONENT value: '{TARGET_COMPONENT}'.  Use 'auto', 'transformer', or 'unet'.")
    raise SystemExit(1)

info(f"Quantizing pipeline component: {Fore.YELLOW}{component_name}{Style.RESET_ALL}")

# --- Apply quantization to main diffusion component ---
info("Applying SDNQ quantization (this may take several minutes) ...")
quantized_component = sdnq_post_load_quant(
    component,
    weights_dtype=QUANT_DTYPE,
    group_size=GROUP_SIZE,
    svd_rank=SVD_RANK,
    svd_steps=SVD_STEPS,
    use_svd=USE_SVD,
    quant_conv=QUANT_CONV,
    quant_embedding=QUANT_EMBEDDING,
    use_dynamic_quantization=USE_DYNAMIC,
    dequantize_fp32=DEQUANTIZE_FP32,
    modules_to_not_convert=MODULES_TO_NOT_CONVERT,
    quantization_device=QUANTIZATION_DEVICE,
    return_device=RETURN_DEVICE,
)
setattr(pipe, component_name, quantized_component)
success(f"{component_name} quantization complete.")

# --- Quantize text encoder(s) ---
if QUANT_TEXT_ENCODER:
    te_attr_names = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    found_any = False
    for te_name in te_attr_names:
        te = getattr(pipe, te_name, None)
        if te is None:
            continue
        found_any = True
        info(f"Quantizing {te_name} (dtype={TEXT_ENCODER_QUANT_DTYPE}) ...")
        quantized_te = sdnq_post_load_quant(
            te,
            weights_dtype=TEXT_ENCODER_QUANT_DTYPE,
            group_size=GROUP_SIZE,
            use_svd=False,          # SVD rarely beneficial for text encoders
            quant_conv=False,
            quant_embedding=False,
            use_dynamic_quantization=False,
            dequantize_fp32=DEQUANTIZE_FP32,
            modules_to_not_convert=MODULES_TO_NOT_CONVERT,
            quantization_device=QUANTIZATION_DEVICE,
            return_device=RETURN_DEVICE,
        )
        setattr(pipe, te_name, quantized_te)
        success(f"{te_name} quantization complete.")
    if not found_any:
        warn("QUANT_TEXT_ENCODER=True but no text_encoder attribute found on pipeline — skipping.")

# --- Save ---
os.makedirs(output_path, exist_ok=True)
info(f"Saving quantized pipeline to {output_path} ...")
pipe.save_pretrained(output_path)
success(f"Saved to: {output_path}")

print(f"\n{Fore.MAGENTA}Load the quantized model in SD.Next by pointing to:{Style.RESET_ALL}")
print(f"  {output_path}")
print(f"\n{Fore.CYAN}Quantized component : {component_name}")
print(f"Dtype               : {QUANT_DTYPE}")
if USE_SVD:
    print(f"SVD rank            : {SVD_RANK}")
if QUANT_TEXT_ENCODER:
    print(f"Text encoder dtype  : {TEXT_ENCODER_QUANT_DTYPE}")
print(f"Output folder       : {output_name}{Style.RESET_ALL}")
