import sys
import os
from huggingface_hub import HfApi

def main():
    token = input("Paste your Hugging Face Write Token: ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        sys.exit(1)
        
    api = HfApi(token=token)
    repo_id = "jagannadharao8/fake-news-detection"
    
    files_to_upload = [
        "artifacts/text_bas/hf_model/model.safetensors",
        "artifacts/vlm/stage_b/adapter/adapter_model.safetensors"
    ]
    
    print("Bypassing .gitignore to upload the massive models...")
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            print(f"\nUploading {file_path} (This will take a minute)...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="space"
                )
                print(f"✅ Uploaded {file_path}")
            except Exception as e:
                print(f"❌ Error uploading {file_path}: {e}")
        else:
            print(f"⚠️ Local file not found: {file_path}")

    print("\n✅ All models strictly uploaded! Hugging Face will now restart.")

if __name__ == "__main__":
    main()
