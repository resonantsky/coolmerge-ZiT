import os, re, shutil, torch, json
import safetensors.torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from colorama import Fore, Style, init

# SD.Next support, this runs in the SD.Next Environment,
# You can convert civitai Z-Image Turbo safetensors checkpoints to diffusers format with this script.

# Initialize colorama (auto-reset so colors don't bleed)
init(autoreset=True)

HF_TOKEN        = "your hugging face token"
SOURCE_CKPT     = "the safetensors checkpoint you downloaded"
OUTPUT_DIR      = "path for this folder structure: transformer/, vae/, text_encoder/"
HF_REPO         = "Tongyi-MAI/Z-Image-Turbo" # The Hugging Face repo to pull from for the VAE and text encoder etc; 
HF_CACHE        = "Your local hugging face cache folder"
LOCAL_VAE       = "A local vae override"
LOCAL_TEXT_ENCODER = "A local text encoder override"
TORCH_DTYPE     = None  # Set to torch.float16, torch.bfloat16, or torch.float32 to cast tensors; None = no casting (preserve source dtype)

Z_IMAGE_RENAME = {
    "final_layer.":         "all_final_layer.2-1.",
    "x_embedder.":          "all_x_embedder.2-1.",
    ".attention.out.bias":  ".attention.to_out.0.bias",
    ".attention.k_norm.weight": ".attention.norm_k.weight",
    ".attention.q_norm.weight": ".attention.norm_q.weight",
    ".attention.out.weight": ".attention.to_out.0.weight",
    "model.diffusion_model.": "",
}

def convert_vae_state_dict_to_diffusers(old_sd):
    """Remap taming-transformers/LDM VAE keys to diffusers AutoencoderKL keys.
    decoder up.i → up_blocks.(NUM_UP-1-i) to match diffusers reversed ordering.
    Also reshapes attention weights from 4D [C,C,1,1] (conv) to 2D [C,C] (linear).
    """
    NUM_UP = 4
    new_sd = {}
    for key, val in old_sd.items():
        k = key
        # encoder down blocks
        k = re.sub(r'^(encoder)\.down\.(\d+)\.block\.(\d+)\.nin_shortcut',
                   lambda m: f'{m.group(1)}.down_blocks.{m.group(2)}.resnets.{m.group(3)}.conv_shortcut', k)
        k = re.sub(r'^(encoder)\.down\.(\d+)\.block\.(\d+)\.',
                   lambda m: f'{m.group(1)}.down_blocks.{m.group(2)}.resnets.{m.group(3)}.', k)
        k = re.sub(r'^(encoder)\.down\.(\d+)\.downsample\.conv',
                   lambda m: f'{m.group(1)}.down_blocks.{m.group(2)}.downsamplers.0.conv', k)
        # decoder up blocks (old index i → diffusers index NUM_UP-1-i)
        k = re.sub(r'^(decoder)\.up\.(\d+)\.block\.(\d+)\.nin_shortcut',
                   lambda m: f'{m.group(1)}.up_blocks.{NUM_UP-1-int(m.group(2))}.resnets.{m.group(3)}.conv_shortcut', k)
        k = re.sub(r'^(decoder)\.up\.(\d+)\.block\.(\d+)\.',
                   lambda m: f'{m.group(1)}.up_blocks.{NUM_UP-1-int(m.group(2))}.resnets.{m.group(3)}.', k)
        k = re.sub(r'^(decoder)\.up\.(\d+)\.upsample\.conv',
                   lambda m: f'{m.group(1)}.up_blocks.{NUM_UP-1-int(m.group(2))}.upsamplers.0.conv', k)
        # mid block resnets: block_1 → resnets.0, block_2 → resnets.1
        k = re.sub(r'^(encoder|decoder)\.mid\.block_(\d+)\.',
                   lambda m: f'{m.group(1)}.mid_block.resnets.{int(m.group(2))-1}.', k)
        # mid block attention projections
        k = re.sub(r'^(encoder|decoder)\.mid\.attn_1\.norm',
                   lambda m: f'{m.group(1)}.mid_block.attentions.0.group_norm', k)
        k = re.sub(r'^(encoder|decoder)\.mid\.attn_1\.q\b',
                   lambda m: f'{m.group(1)}.mid_block.attentions.0.to_q', k)
        k = re.sub(r'^(encoder|decoder)\.mid\.attn_1\.k\b',
                   lambda m: f'{m.group(1)}.mid_block.attentions.0.to_k', k)
        k = re.sub(r'^(encoder|decoder)\.mid\.attn_1\.v\b',
                   lambda m: f'{m.group(1)}.mid_block.attentions.0.to_v', k)
        k = re.sub(r'^(encoder|decoder)\.mid\.attn_1\.proj_out',
                   lambda m: f'{m.group(1)}.mid_block.attentions.0.to_out.0', k)
        # norm_out → conv_norm_out
        k = re.sub(r'^(encoder|decoder)\.norm_out',
                   lambda m: f'{m.group(1)}.conv_norm_out', k)
        # Attention weight reshape: 4D [C,C,1,1] → 2D [C,C] (no dtype change)
        if k != key and 'mid_block.attentions' in k and k.endswith('.weight') and val.ndim == 4:
            val = val.squeeze(-1).squeeze(-1)
        new_sd[k] = val
    return new_sd


def generate_vae_config_from_state(state_dict, output_path):
    in_channels  = state_dict.get("encoder.conv_in.weight", None)
    out_channels = state_dict.get("decoder.conv_out.weight", None)
    latent_channels = state_dict.get("decoder.conv_in.weight", None)

    in_channels     = in_channels.shape[1]     if in_channels     is not None else 3
    out_channels    = out_channels.shape[0]    if out_channels    is not None else 3
    latent_channels = latent_channels.shape[1] if latent_channels is not None else 4

    block_out_channels = []
    for i in range(4):
        key = f"encoder.down.{i}.block.0.conv1.weight"
        if key in state_dict:
            block_out_channels.append(state_dict[key].shape[0])

    config = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.36.0.dev0",
        "act_fn": "silu",
        "block_out_channels": block_out_channels,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "layers_per_block": 2,
        "mid_block_add_attention": True,
        "norm_num_groups": 32,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "latent_channels": latent_channels,
        "sample_size": 1024,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "use_post_quant_conv": False,
        "use_quant_conv": False,
        "force_upcast": False,
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    success(f"Wrote VAE config.json to {output_path}")


DROP_KEYS = {"norm_final.weight"}

# --- Console helpers ---
def info(msg):
    print(f"\n{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")

def success(msg):
    print(f"\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {msg}")

def warn(msg):
    print(f"\n{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")

def error(msg):
    print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

_DTYPE_NAMES = {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}

def detect_dtype(path, sample=20):
    dtypes = set()
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in list(f.keys())[:sample]:
            dtypes.add(f.get_tensor(k).dtype)
    if len(dtypes) == 1:
        d = next(iter(dtypes))
        return d, _DTYPE_NAMES.get(d, str(d))
    names = [_DTYPE_NAMES.get(d, str(d)) for d in sorted(dtypes, key=str)]
    return dtypes, f"MIXED ({', '.join(names)})"

def maybe_cast(tensor):
    if TORCH_DTYPE is not None:
        return tensor.to(TORCH_DTYPE).clone()
    return tensor.clone()

# --- Functions ---
def rename_transformer_keys(state_dict):
    result = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old, new in Z_IMAGE_RENAME.items():
            new_key = new_key.replace(old, new)
        if new_key in DROP_KEYS:
            continue
        if ".attention.qkv.weight" in new_key:
            q, k, v = torch.chunk(tensor, 3, dim=0)
            result[new_key.replace(".attention.qkv.weight", ".attention.to_q.weight")] = q
            result[new_key.replace(".attention.qkv.weight", ".attention.to_k.weight")] = k
            result[new_key.replace(".attention.qkv.weight", ".attention.to_v.weight")] = v
        else:
            result[new_key] = tensor
    return result

def validate_output(path, name="component"):
    check = safetensors.torch.load_file(path, device="cpu")
    if not check:
        warn(f"{name}: no tensors found")
        return path
    dtypes = {v.dtype for v in check.values()}
    if len(dtypes) == 1:
        d = next(iter(dtypes))
        success(f"{name} saved as {_DTYPE_NAMES.get(d, str(d))}")
    else:
        names = [_DTYPE_NAMES.get(d, str(d)) for d in sorted(dtypes, key=str)]
        warn(f"{name} has mixed dtypes: {', '.join(names)}")
    return path

# --- Workflow ---
info(f"Downloading HF repo {HF_REPO} ...")
hf_snapshot = snapshot_download(
    repo_id=HF_REPO,
    cache_dir=HF_CACHE,
    token=HF_TOKEN,
    ignore_patterns=[
        "transformer/*.safetensors",
        "transformer/*.bin",
        "transformer/*.index.json",
    ],
)
success(f"HF snapshot downloaded: {hf_snapshot}")

info(f"Copying snapshot to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
for item in os.listdir(hf_snapshot):
    src = os.path.join(hf_snapshot, item)
    dst = os.path.join(OUTPUT_DIR, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
success("Snapshot copied successfully.")

# Purge any transformer weight/index files the HF snapshot placed — SOURCE_CKPT is the sole provider
_xfm_snapshot_dir = os.path.join(OUTPUT_DIR, "transformer")
if os.path.isdir(_xfm_snapshot_dir):
    _purged = [fn for fn in os.listdir(_xfm_snapshot_dir)
               if fn.endswith(".safetensors") or fn.endswith(".bin") or fn.endswith(".index.json")]
    for _fn in _purged:
        os.remove(os.path.join(_xfm_snapshot_dir, _fn))
    if _purged:
        info(f"Cleared {len(_purged)} HF transformer file(s) from output — SOURCE_CKPT weights will be used instead.")

info(f"Loading source checkpoint: {SOURCE_CKPT} ...")
_src_dtype, _src_dtype_name = detect_dtype(SOURCE_CKPT)
info(f"Source checkpoint dtype detected: {Fore.YELLOW}{_src_dtype_name}{Style.RESET_ALL}")
if TORCH_DTYPE is not None:
    info(f"TORCH_DTYPE is set — tensors will be cast to {_DTYPE_NAMES.get(TORCH_DTYPE, str(TORCH_DTYPE))}")
else:
    info("TORCH_DTYPE is None — no casting, source dtype preserved")
raw = {}
with safe_open(SOURCE_CKPT, framework="pt", device="cpu") as f:
    for k in f.keys():
        raw[k] = maybe_cast(f.get_tensor(k))
success("Source checkpoint loaded.")

converted = rename_transformer_keys(raw)
if TORCH_DTYPE is not None:
    converted = {k: v.to(TORCH_DTYPE) for k, v in converted.items()}

transformer_dir = os.path.join(OUTPUT_DIR, "transformer")
os.makedirs(transformer_dir, exist_ok=True)
out_weights = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
safetensors.torch.save_file(converted, out_weights)

info("Validating transformer weights ...")
validate_output(out_weights, "transformer")

vae_dir = os.path.join(OUTPUT_DIR, "vae")
os.makedirs(vae_dir, exist_ok=True)
vae_out = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")

if LOCAL_VAE and os.path.exists(LOCAL_VAE):
    info(f"Using local VAE override: {LOCAL_VAE}")
    _vae_dtype, _vae_dtype_name = detect_dtype(LOCAL_VAE)
    info(f"VAE dtype detected: {Fore.YELLOW}{_vae_dtype_name}{Style.RESET_ALL}")
    vae_raw = {}
    with safe_open(LOCAL_VAE, framework="pt", device="cpu") as f:
        for k in f.keys():
            vae_raw[k] = maybe_cast(f.get_tensor(k))

    # Generate config.json from original (pre-remap) state dict
    generate_vae_config_from_state(vae_raw, os.path.join(vae_dir, "config.json"))

    converted_vae = convert_vae_state_dict_to_diffusers(vae_raw)
    info(f"Converted {len(vae_raw)} VAE keys → {len(converted_vae)} diffusers keys")

    safetensors.torch.save_file(converted_vae, vae_out)
    validate_output(vae_out, "VAE")
else:
    warn("Keeping HF repo VAE (may be fp32). Consider overriding with LOCAL_VAE.")

# Text Encoder
text_dir = os.path.join(OUTPUT_DIR, "text_encoder")
os.makedirs(text_dir, exist_ok=True)
text_out = os.path.join(text_dir, "model.safetensors")

if LOCAL_TEXT_ENCODER and os.path.exists(LOCAL_TEXT_ENCODER):
    info(f"Using local text encoder: {LOCAL_TEXT_ENCODER}")
    _te_dtype, _te_dtype_name = detect_dtype(LOCAL_TEXT_ENCODER)
    info(f"Text encoder dtype detected: {Fore.YELLOW}{_te_dtype_name}{Style.RESET_ALL}")
    te_raw = {}
    with safe_open(LOCAL_TEXT_ENCODER, framework="pt", device="cpu") as f:
        for k in f.keys():
            te_raw[k] = maybe_cast(f.get_tensor(k))
    safetensors.torch.save_file(te_raw, text_out)
    validate_output(text_out, "text_encoder")
elif os.path.exists(text_out):
    warn(f"LOCAL_TEXT_ENCODER not found — using HF snapshot text encoder from {text_out}")
    validate_output(text_out, "text_encoder")
else:
    warn("No text encoder found (local or HF snapshot) — pipeline may fall back to HF hub at runtime.")

index_file = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
if os.path.exists(index_file):
    os.remove(index_file)
    info("Removed transformer shard index file.")

success(f"\nDone. Output folder: {OUTPUT_DIR}")
print(f"{Fore.MAGENTA}Load it in SD.Next by pointing to this folder (or add to Diffusers models list).{Style.RESET_ALL}")
if LOCAL_TEXT_ENCODER and os.path.exists(LOCAL_TEXT_ENCODER):
    info(f"Using local text encoder: {LOCAL_TEXT_ENCODER}")
    _te_dtype, _te_dtype_name = detect_dtype(LOCAL_TEXT_ENCODER)
    info(f"Text encoder dtype detected: {Fore.YELLOW}{_te_dtype_name}{Style.RESET_ALL}")
    te_raw = {}
    with safe_open(LOCAL_TEXT_ENCODER, framework="pt", device="cpu") as f:
        for k in f.keys():
            te_raw[k] = maybe_cast(f.get_tensor(k))
    safetensors.torch.save_file(te_raw, text_out)
    validate_output(text_out, "text_encoder")
elif os.path.exists(text_out):
    warn(f"LOCAL_TEXT_ENCODER not found — using HF snapshot text encoder from {text_out}")
    validate_output(text_out, "text_encoder")
else:
    warn("No text encoder found (local or HF snapshot) — pipeline may fall back to HF hub at runtime.")

index_file = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
if os.path.exists(index_file):
    os.remove(index_file)
    info("Removed transformer shard index file.")

success(f"\nDone. Output folder: {OUTPUT_DIR}")
print(f"{Fore.MAGENTA}Load it in SD.Next by pointing to this folder (or add to Diffusers models list).{Style.RESET_ALL}")
