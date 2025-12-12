import os
from huggingface_hub import snapshot_download

MODELS = {
    'slm': 'ibm-granite/granite-3.0-1b-a400m-instruct',
    'llm': 'microsoft/Phi-3-mini-4k-instruct'
}

TARGET_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

os.makedirs(TARGET_DIR, exist_ok=True)

for key, repo_id in MODELS.items():
    print(f"Downloading {key} from {repo_id} ...")
    out_dir = os.path.join(TARGET_DIR, key)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"Skipped {key}: already exists at {out_dir}")
        continue
    try:
        snapshot_download(repo_id, cache_dir=out_dir, local_dir=out_dir, allow_patterns=None)
        print(f"Downloaded {key} -> {out_dir}")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

print("Done. Models are stored under the 'models' directory.")
