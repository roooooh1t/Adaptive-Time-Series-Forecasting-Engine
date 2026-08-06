import kagglehub
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env") # By default reads environment variables from ".env" only

os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN") # Load the kaggle token

kagglehub.login()

print("HELLO")