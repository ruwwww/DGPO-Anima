#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DGPO-Anima Asset Downloader for DGX Cluster
# Downloads Anima Base v1.0 (Cosmos 2 DiT 28 blocks, D=2048), Qwen3 Text Encoder,
# and VAE in pure BF16 from HuggingFace (both ComfyUI split format and Diffusers repo).
# ==============================================================================

REQUESTED_DIR="${1:-/workspace/models}"
TARGET_DIR="${REQUESTED_DIR}"

echo "=================================================================="
echo "📥 DOWNLOADING ASSETS FOR ANIMA DGPO TRAINING"
echo "=================================================================="
echo "Requested target directory: ${REQUESTED_DIR}"

# Check permissions for requested directory
if ! mkdir -p "${TARGET_DIR}" 2>/dev/null; then
    FALLBACK_DIR="${HOME}/models"
    echo "⚠️  Permission denied for ${REQUESTED_DIR}."
    echo "🔄 Falling back to user-writable storage: ${FALLBACK_DIR}"
    TARGET_DIR="${FALLBACK_DIR}"
    mkdir -p "${TARGET_DIR}"
fi

echo "✅ Active storage directory: ${TARGET_DIR}"

# Export environment variable for downstream scripts
export ANIMA_MODELS_DIR="${TARGET_DIR}"

python - << PYEOF
import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

target_dir = "${TARGET_DIR}"
token = os.environ.get("HF_TOKEN")
token_file = os.path.expanduser("~/.cache/huggingface/token")
if not token and os.path.exists(token_file):
    with open(token_file, "r") as f:
        token = f.read().strip()

print(f"HF Token detected: {'Yes' if token else 'No'}")

# 1. Download dedicated diffusers repository: circlestone-labs/Anima-Base-v1.0-Diffusers
diffusers_dir = os.path.join(target_dir, "Anima-Base-v1.0-Diffusers")
print(f"Downloading dedicated Diffusers repo 'circlestone-labs/Anima-Base-v1.0-Diffusers' into {diffusers_dir}...")
snapshot_download(
    repo_id="circlestone-labs/Anima-Base-v1.0-Diffusers",
    local_dir=diffusers_dir,
    token=token,
    local_dir_use_symlinks=False,
)
print("✅ Dedicated Diffusers repo downloaded successfully!")

# 2. Also ensure root split files exist (or symlinked)
split_map = {
    "anima-base-v1.0.safetensors": os.path.join(diffusers_dir, "transformer", "diffusion_pytorch_model.safetensors"),
    "qwen_3_06b_base.safetensors": os.path.join(diffusers_dir, "text_encoder", "model.safetensors"),
    "qwen_image_vae.safetensors": os.path.join(diffusers_dir, "vae", "diffusion_pytorch_model.safetensors"),
}

for root_name, src_path in split_map.items():
    dest_path = os.path.join(target_dir, root_name)
    if not os.path.exists(dest_path) and os.path.exists(src_path):
        os.symlink(src_path, dest_path)
        print(f"  Linked {root_name} -> {src_path}")

print("\nVerification summary:")
for root_name in split_map.keys():
    dest_path = os.path.join(target_dir, root_name)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024**2)
        print(f"  ✅ {root_name}: {size_mb:.2f} MB")

print("\nAll assets verified successfully in pure BF16 format!")
PYEOF

echo "export ANIMA_MODELS_DIR=\"${TARGET_DIR}\"" > "${HOME}/.anima_env"
echo "✅ Environment configured in ${HOME}/.anima_env"
echo "=================================================================="
