import os
from pathlib import Path
from dotenv import load_dotenv
import kagglehub

def download_m5_data():
    # 1. Load the environment variables from the .env file
    # This resolves the path to the root folder where .env lives
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / '.env'
    
    # Load the token into the system environment
    load_dotenv(dotenv_path=env_path)
    
    # 2. Define the target directory for the raw data
    raw_data_dir = project_root / 'data' / 'raw'
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Target directory set to: {raw_data_dir}")
    print("Authenticating and downloading dataset...")
    
    # 3. Download using kagglehub
    # Because KAGGLE_API_TOKEN is now in the environment, kagglehub authenticates automatically
    dataset_path = kagglehub.competition_download(
        "m5-forecasting-accuracy", 
        output_dir=str(raw_data_dir)
    )
    
    print(f"\nSuccess! Competition files have been downloaded and extracted to: {dataset_path}")

if __name__ == "__main__":
    download_m5_data()