import os
from pathlib import Path
from dotenv import load_dotenv

def download_kaggle_dataset():
    load_dotenv()
    dataset = os.getenv("KAGGLE_DATASET_SLUG")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[download] Downloading from Kaggle...")
    os.system(f'kaggle datasets download -d {dataset} -p {raw_dir}')

    print("[download] Unzipping file...")
    os.system(f'powershell -command "Expand-Archive -Force {raw_dir}\\*.zip {raw_dir}"')