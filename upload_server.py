import sys
from huggingface_hub import HfApi

def main():
    token = input("Paste your Hugging Face Write Token: ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        sys.exit(1)
        
    api = HfApi(token=token)
    repo_id = "jagannadharao8/fake-news-detection"
    
    print(f"\nUploading server.py fix to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj="server.py",
            path_in_repo="server.py",
            repo_id=repo_id,
            repo_type="space"
        )
        print("\n✅ Upload complete! Hugging Face will now restart.")
    except Exception as e:
        print(f"\n❌ Error uploading file: {e}")

if __name__ == "__main__":
    main()
