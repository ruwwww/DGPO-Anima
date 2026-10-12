#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Upload Trained LoRA Adapter to HuggingFace Hub
# Usage: bash scripts/upload_lora_hf.sh <ckpt_path> [repo_id]
# ==============================================================================

CKPT_PATH="${1:-}"
REPO_ID="${2:-ruwwww/Anima-DGPO-Text-Rendering}"

if [ -z "${CKPT_PATH}" ]; then
    echo "❌ Error: Checkpoint path is required."
    echo "Usage: bash scripts/upload_lora_hf.sh <ckpt_path> [repo_id]"
    exit 1
fi

if [ ! -e "${CKPT_PATH}" ]; then
    echo "❌ Error: Checkpoint path '${CKPT_PATH}' does not exist."
    exit 1
fi

echo "=================================================================="
echo "🚀 UPLOADING LORA ADAPTER TO HUGGINGFACE HUB"
echo "=================================================================="
echo "Source checkpoint: ${CKPT_PATH}"
echo "Target repository: ${REPO_ID}"

python - << PYEOF
import os
import sys
from huggingface_hub import HfApi

ckpt_path = "${CKPT_PATH}"
repo_id = "${REPO_ID}"

token = os.environ.get("HF_TOKEN")
token_file = os.path.expanduser("~/.cache/huggingface/token")
if not token and os.path.exists(token_file):
    with open(token_file, "r") as f:
        token = f.read().strip()

if not token:
    print("❌ Error: No Hugging Face token found. Run restore_session.sh or set HF_TOKEN.")
    sys.exit(1)

api = HfApi(token=token)
print(f"Ensuring repository '{repo_id}' exists...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

if os.path.isdir(ckpt_path):
    print(f"Uploading directory '{ckpt_path}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=ckpt_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload DGPO-Anima LoRA adapter for Text Rendering",
    )
else:
    filename = os.path.basename(ckpt_path)
    print(f"Uploading file '{ckpt_path}' as '{filename}' to '{repo_id}'...")
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {filename}",
    )

print(f"✅ Checkpoint successfully uploaded to https://huggingface.co/{repo_id}")
PYEOF

echo "=================================================================="
echo "✅ UPLOAD COMPLETE"
echo "=================================================================="
