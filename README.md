# coolmerge-ZiT

Converts a [CivitAI Z-Image Turbo](https://civitai.com/models/) single-file `.safetensors` checkpoint into a diffusers-compatible folder structure ready to load in **SD.Next** (or any Hugging Face Diffusers pipeline).

---

## What it does

1. **Downloads the HF repo** (`Tongyi-MAI/Z-Image-Turbo`) to pull configs, tokenizer, scheduler, and any non-transformer weights — transformer weights are intentionally excluded since `SOURCE_CKPT` is the authoritative source.
2. **Converts the transformer** — renames Z-Image internal keys to diffusers conventions, splits fused QKV projections into separate `to_q` / `to_k` / `to_v` tensors, and drops unused keys (`norm_final.weight`).
3. **Converts the VAE** (optional local override) — remaps taming-transformers / LDM VAE keys to diffusers `AutoencoderKL` layout, reverses decoder block ordering, reshapes conv attention weights to linear, and auto-generates `config.json` from the weight shapes.
4. **Copies the text encoder** (optional local override) — falls back to the HF snapshot encoder if no local path is provided.
5. **Validates** every component on save, reporting dtypes and warning on mixed-precision artifacts.

Output folder structure:

```
OUTPUT_DIR/
├── transformer/
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── diffusion_pytorch_model.safetensors
│   └── config.json
├── text_encoder/
│   └── model.safetensors
├── scheduler/        ← from HF snapshot
├── tokenizer/        ← from HF snapshot
└── model_index.json  ← from HF snapshot
```

---

## Requirements

Runs inside the **SD.Next venv** (all dependencies already present):

```
torch
safetensors
huggingface_hub
colorama
```

---

## Configuration

Edit the constants at the top of the script before running:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your Hugging Face access token |
| `SOURCE_CKPT` | Path to the CivitAI Z-Image Turbo `.safetensors` file |
| `OUTPUT_DIR` | Destination folder for the converted diffusers model |
| `HF_REPO` | HF repo to pull from (default: `Tongyi-MAI/Z-Image-Turbo`) |
| `HF_CACHE` | Your local HF cache directory |
| `LOCAL_VAE` | *(optional)* Path to a local VAE `.safetensors` to use instead of the repo VAE |
| `LOCAL_TEXT_ENCODER` | *(optional)* Path to a local text encoder `.safetensors` |
| `TORCH_DTYPE` | Cast all tensors on save — `torch.float16`, `torch.bfloat16`, `torch.float32`, or `None` to preserve source dtype |

---

## Usage

```powershell
# Activate the SD.Next venv first
& venv\Scripts\Activate.ps1

python coolmerge-ZiT.py
```

On success the script prints the output folder path and a reminder to add it to SD.Next's Diffusers models list.

---

## Key remappings (transformer)

| Source key pattern | Diffusers key pattern |
|---|---|
| `model.diffusion_model.*` | `*` (prefix stripped) |
| `final_layer.*` | `all_final_layer.2-1.*` |
| `x_embedder.*` | `all_x_embedder.2-1.*` |
| `*.attention.qkv.weight` | split → `to_q` / `to_k` / `to_v` |
| `*.attention.out.*` | `*.attention.to_out.0.*` |
| `*.attention.{q,k}_norm.weight` | `*.attention.norm_{q,k}.weight` |
| `norm_final.weight` | dropped |

---

## Notes

- The VAE block index order is **reversed** between LDM and diffusers: `up.i` → `up_blocks.(NUM_UP-1-i)`. This is handled automatically.
- If `LOCAL_VAE` is not set, the VAE from the HF snapshot is used as-is.
- Tested against `Tongyi-MAI/Z-Image-Turbo` rev as of early 2025.
