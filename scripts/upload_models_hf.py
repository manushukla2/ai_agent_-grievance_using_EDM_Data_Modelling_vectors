"""
Upload models to Hugging Face Hub for storage.
This avoids GitHub's 2GB file size limit.
"""
from huggingface_hub import HfApi, create_repo, upload_folder
import os

HF_USERNAME = "your-username"  # Change this
REPO_NAME = "grievance-rag-models"

def upload_models():
    api = HfApi()
    
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository created/exists: {repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    print("Uploading SLM model...")
    upload_folder(
        folder_path=os.path.join(models_path, 'slm'),
        repo_id=repo_id,
        path_in_repo="slm",
        repo_type="model"
    )
    print("SLM uploaded!")
    
    print("Uploading LLM model...")
    upload_folder(
        folder_path=os.path.join(models_path, 'llm'),
        repo_id=repo_id,
        path_in_repo="llm",
        repo_type="model"
    )
    print("LLM uploaded!")
    
    print(f"\nModels uploaded to: https://huggingface.co/{repo_id}")
    print("Update download_models.py to download from this repo")

if __name__ == "__main__":
    print("First login to Hugging Face:")
    print("Run: huggingface-cli login")
    print("\nThen update HF_USERNAME in this script and run again")
    upload_models()
