"""
Uploads rag_sft_dataset.jsonl to Adaption Labs via the official Python SDK.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from adaption import Adaption

DATASET_FILE = Path("rag_sft_dataset.jsonl")
if not DATASET_FILE.exists():
    DATASET_FILE = Path(config.DATA_DIR) / "processed" / "rag_sft_dataset.jsonl"


def main():
    api_key = config.ADAPTION_API_KEY or os.getenv("ADAPTION_API_KEY", "")
    if not api_key:
        print("Error: ADAPTION_API_KEY is missing from .env")
        return

    print(f"Connecting to Adaption Labs with API Key: {api_key[:10]}...{api_key[-4:]}")
    client = Adaption(api_key=api_key)

    if not DATASET_FILE.exists():
        print(f"Error: Dataset file not found at {DATASET_FILE}")
        return

    file_size_mb = DATASET_FILE.stat().st_size / (1024 * 1024)
    print(f"Uploading '{DATASET_FILE.name}' ({file_size_mb:.2f} MB) to Adaption Labs...")

    try:
        response = client.datasets.upload_file(
            path=str(DATASET_FILE),
            name="indic_multilingual_voice_rag",
        )
        print("\n Upload Successful!")
        print(f"Dataset Response: {response}")
        print("\nYou can now see and train this dataset in your Adaption Labs dashboard at:")
        print("🔗 https://adaptionlabs.ai/app/datasets")
    except Exception as e:
        print(f"\nUpload error: {e}")


if __name__ == "__main__":
    main()
