import sys
from huggingface_hub import HfApi

def main():
    print("=== Hugging Face Model Uploader ===")
    token = input("Paste your Hugging Face Write Token (invisible while typing): ").strip()
    
    if not token:
        print("Error: Token cannot be empty.")
        sys.exit(1)
        
    api = HfApi(token=token)
    
    repo_id = "jagannadharao8/fake-news-detection"
    print(f"\nConnecting to {repo_id}...")
    
    try:
        # We only need to upload the specific folders required for inference
        print("Uploading ML Models to your Cloud Server. This may take a few minutes depending on your internet speed...")
        
        api.upload_folder(
            folder_path="artifacts",
            repo_id=repo_id,
            repo_type="space",
            path_in_repo="artifacts"
        )
        print("\n✅ Upload complete! Hugging Face will now automatically restart your server with the models loaded.")
    except Exception as e:
        print(f"\n❌ Error uploading models: {e}")

if __name__ == "__main__":
    main()
