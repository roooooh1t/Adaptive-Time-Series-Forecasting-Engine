import os
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path
import os

# Get Absolute Path of this file 
BASE_DIR = Path(__file__).resolve().parent 

# By default reads environment variables from ".env" only
load_dotenv(dotenv_path=".env") 

# Load the kaggle token
os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN") 

# Download the actual m5 dataset directly from kaggle (You must have joined this compitition to be able to download (Else will get 403 error))
kagglehub.competition_download("m5-forecasting-accuracy", output_dir=f"{BASE_DIR}/../../data") 